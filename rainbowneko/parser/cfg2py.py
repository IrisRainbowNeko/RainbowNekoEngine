import ast
import inspect
import types
from collections import defaultdict
from pathlib import Path
from functools import partial

from yapf.yapflib.yapf_api import FormatCode

from rainbowneko.utils import is_valid_variable_name, is_dict, is_list


class LambdaFinder(ast.NodeVisitor):
    def __init__(self):
        self.lambda_list = []

    def visit_Lambda(self, node):
        self.lambda_list.append(node)
        self.generic_visit(node)


class ConfigCodeReconstructor:
    global_handlers = {}

    def __init__(self):
        self.type_handlers = {}
        self.required_imports = defaultdict(set)
        self.seen_types = set()

        # Register basic handlers
        self.register_type_handler('_custom_object', self._handle_custom_object)
        self.register_type_handler('_dict', self._dict_to_ast)
        self.register_type_handler('_list', self._list_to_ast)
        self.register_type_handler('Path', self._handle_path)
        self.register_type_handler('_module', self._handle_module)
        self.register_type_handler('_method', self._handle_method)
        self.register_type_handler('_class_instance', self._handle_class_instance)
        self.register_type_handler('_object_instance', self._handle_object_instance)
        for name, handler in self.global_handlers.items():
            self.register_type_handler(name, handler)

    def register_type_handler(self, type_name, handler_func):
        """Register type handler"""
        self.type_handlers[type_name] = handler_func

    def _handle_path(self, obj, **kwargs):
        """Handle Path objects"""
        if isinstance(obj, Path):
            self.required_imports['pathlib'].add('Path')
        else:
            return False

        path = str(obj)
        # Create Path object
        path_expr = ast.Call(
            func=ast.Name(id='Path', ctx=ast.Load()),
            args=[ast.Constant(value=path)],
            keywords=[]
        )

        return path_expr

    def _handle_custom_object(self, obj, **kwargs):
        """Handle dictionary-form custom objects"""
        if is_dict(obj) and '_target_' in obj:
            type_name = obj['_target_']
        else:
            return False

        obj_data = {k: v for k, v in obj.items() if k != '_target_'}

        args = []
        keywords = []

        for k, v in obj_data.items():
            if k == '_args_':
                for arg in v:
                    args.append(self._value_to_ast(arg))
            else:
                keywords.append(ast.keyword(arg=k, value=self._value_to_ast(v, k)))

        if type_name == getattr and len(args) == 2:
            return ast.Attribute(value=args[0], attr=args[1].value, ctx=ast.Load())
        elif callable(type_name) or is_dict(type_name):
            return ast.Call(
                func=self._value_to_ast(type_name),
                args=args,
                keywords=keywords
            )
        else: # str
            # Extract class name (handle possible module path)
            if '.' in type_name:
                parts = type_name.split('.')
                module_path = '.'.join(parts[:-1])
                class_name = parts[-1]
                self.required_imports[module_path].add(class_name)
            else:
                class_name = type_name

            return ast.Call(
                func=ast.Name(id=class_name, ctx=ast.Load()),
                args=args,
                keywords=keywords
            )

    def _handle_method(self, obj, **kwargs):
        if isinstance(obj, (types.MethodType, types.FunctionType)):
            module_name = obj.__module__
            if module_name == '__main__' or module_name == 'builtins':
                module_name = None
        else:
            return False

        if isinstance(obj, types.BuiltinMethodType):
            method_name = obj.__name__
            self.required_imports[module_name].add(method_name)
            return ast.Name(id=method_name, ctx=ast.Load())
        elif isinstance(obj, types.MethodType):
            v = obj.__self__
            method_name = obj.__name__
            return ast.Attribute(value=self._value_to_ast(v), attr=method_name, ctx=ast.Load())
        elif obj.__name__ == '<lambda>':
            try:
                # Check if source code can be obtained
                source = inspect.getsource(obj).strip()
                tree = ast.parse(source)
                finder = LambdaFinder()
                finder.visit(tree)
                if len(finder.lambda_list) > 0:
                    return finder.lambda_list[0]
                else:
                    raise ValueError(f'lambda not found: {source}')
            except:
                # Cannot parse, use placeholder
                lambda_expr = ast.Lambda(
                    args=[],
                    body=ast.Name(id='lambda_expr', ctx=ast.Load())
                )
                return lambda_expr
        else:
            if obj.__name__ != obj.__qualname__: # classmethod or staticmethod
                method_name = obj.__qualname__
                self.required_imports[module_name].add(method_name.split('.')[0])
            else:
                method_name = obj.__name__
                self.required_imports[module_name].add(method_name)

            return ast.Name(id=method_name, ctx=ast.Load())
    
    def _handle_class_instance(self, obj, **kwargs):
        """Handle class instances, treating them as class names"""
        if isinstance(obj, type):
            class_name = obj.__name__
            module_name = obj.__module__

            # Build type identifier to avoid duplicate processing
            type_id = f"{module_name}.{class_name}"

            # If this is a new type, add to seen types and handle imports
            if type_id not in self.seen_types and module_name != '__main__':
                self.seen_types.add(type_id)
                # Add import requirements
                if module_name != 'builtins':
                    self.required_imports[module_name].add(class_name)

            return ast.Name(id=class_name, ctx=ast.Load())
        else:
            return False

    def _handle_object_instance(self, obj, max_tensor_len=100, **kwargs):
        """Handle instantiated objects, treating non-constructor parameters as separate assignment statements"""
        if (obj is not None and
            not isinstance(obj, (int, float, str, bool, dict, list, tuple, set)) and
            not is_dict(obj) and not is_list(obj) and
            hasattr(obj, '__class__') and
            obj.__class__.__module__ != 'builtins'):
            # Automatically identify object type and add imports
            class_type = type(obj)
            class_name = class_type.__name__
            module_name = class_type.__module__

            # Build type identifier to avoid duplicate processing
            type_id = f"{module_name}.{class_name}"

            # If this is a new type, add to seen types and handle imports
            if type_id not in self.seen_types and module_name != '__main__':
                self.seen_types.add(type_id)
                # Add import requirements
                if module_name != 'builtins':
                    self.required_imports[module_name].add(class_name)
        else:
            return False

        try:
            # Check if it's torch.dtype
            if module_name == 'torch':
                if class_name == 'dtype':
                    module_name, class_name = str(obj).rsplit('.', maxsplit=1)
                    return ast.Attribute(value=ast.Name(id=module_name, ctx=ast.Load()), attr=class_name, ctx=ast.Load())

                elif class_name == 'device':
                    return ast.Call(
                        func=ast.Attribute(value=ast.Name(id=module_name, ctx=ast.Load()), attr=class_name, ctx=ast.Load()),
                        args=[
                            ast.Constant(value=obj.type),
                            ast.Constant(value=obj.index),
                        ],
                        keywords=[]
                    )
                elif class_name == 'Tensor':
                    # Handle torch.Tensor objects
                    device = self._value_to_ast(obj.device)
                    dtype = self._value_to_ast(obj.dtype)
                    if obj.numel() > max_tensor_len: # Too large, give up. Such a large tensor may be created automatically.
                        return ast.Name(id='None', ctx=ast.Load())
                    else:
                        return ast.Call(
                            func=ast.Attribute(value=ast.Name(id=module_name, ctx=ast.Load()), attr='tensor', ctx=ast.Load()),
                            args=[self._value_to_ast(obj.tolist())],
                            keywords=[
                                ast.keyword(arg='device', value=device),
                                ast.keyword(arg='dtype', value=dtype)
                            ]
                        )
            
            if isinstance(obj, partial):
                # Handle functools.partial objects
                func = self._value_to_ast(obj.func)
                args = [self._value_to_ast(arg) for arg in obj.args]
                keywords = [ast.keyword(arg='_partial_', value=ast.Constant(value=True))] + \
                           [ast.keyword(arg=k, value=self._value_to_ast(v)) for k, v in obj.keywords.items()]
                return ast.Call(func=func, args=args, keywords=keywords)

            # Check if constructor parameter inspection is supported
            has_init_signature = False
            accepts_kwargs = False
            init_params = {}

            # Try to get constructor signature
            if hasattr(obj.__class__, '__init__'):
                try:
                    init_signature = inspect.signature(obj.__class__.__init__)
                    has_init_signature = True

                    # Check if **kwargs is accepted
                    for param_name, param in init_signature.parameters.items():
                        if param.kind == inspect.Parameter.VAR_KEYWORD:
                            accepts_kwargs = True
                            break

                    # Extract initialization parameters
                    for param_name, param in init_signature.parameters.items():
                        if param_name != 'self' and hasattr(obj, param_name):
                            init_params[param_name] = getattr(obj, param_name)
                except (ValueError, TypeError):
                    # Cannot get signature, fallback to using all attributes
                    has_init_signature = False

            # Get all visible attributes of the object
            all_attrs = {}
            if hasattr(obj, '__dict__'):
                all_attrs.update({k: v for k, v in obj.__dict__.items() if not k.startswith('_') and 
                                  not (isinstance(v, types.MethodType) and not isinstance(v.__self__, type))})

            # Determine which attributes are constructor parameters and which are extra attributes
            constructor_attrs = {}
            extra_attrs = {}

            if not has_init_signature or accepts_kwargs:
                # If constructor signature cannot be obtained or constructor accepts **kwargs,
                # treat all attributes as constructor parameters
                constructor_attrs = all_attrs
            else:
                # Otherwise, only include constructor parameters in the constructor
                constructor_attrs = init_params
                # Remaining attributes as extra attributes
                for attr_name, attr_value in all_attrs.items():
                    if attr_name not in constructor_attrs:
                        extra_attrs[attr_name] = attr_value

            # Create constructor parameters
            keywords = []
            for param_name, param_value in constructor_attrs.items():
                keywords.append(ast.keyword(
                    arg=param_name,
                    value=self._value_to_ast(param_value, param_name)
                ))

            # If there are extra attributes, add them to the parameter list
            if extra_attrs:
                ex_keywords = []
                for param_name, param_value in extra_attrs.items():
                    ex_keywords.append(ast.keyword(
                        arg=param_name,
                        value=self._value_to_ast(param_value, param_name)
                    ))
                keywords.append(
                    ast.keyword(
                        arg='_ex_attrs_',
                        value=ast.Call(
                            func=ast.Name(id='dict', ctx=ast.Load()),
                            args=[],
                            keywords=ex_keywords
                        )
                    )
                )

            # Create constructor call
            constructor_call = ast.Call(
                func=ast.Name(id=class_name, ctx=ast.Load()),
                args=[],
                keywords=keywords
            )

            return constructor_call

        except Exception as e:
            import traceback
            traceback.print_exc()
            # If an error occurs, use simple repr as fallback
            repr_str = repr(obj)
            self.required_imports['direct'].add(f"# Cannot parse object: {repr_str}")
            return ast.Name(id=f"# {repr_str} #", ctx=ast.Load())

    def _handle_module(self, module: types.ModuleType, **kwargs):
        """Handle module objects"""
        if isinstance(module, types.ModuleType):
            parts = module.__name__.split('.')
            module_path = '.'.join(parts[:-1])
            module_name = parts[-1]
            self.required_imports[module_path].add(module_name)
        else:
            return False

        value = parts[-1]
        return ast.Name(id=value, ctx=ast.Load())

    def _value_to_ast(self, value, key=None, parent=None):
        """Convert value to AST node"""
        self.required_imports['rainbowneko.parser'].add('neko_cfg')

        for name, handler in self.type_handlers.items():
            if result := handler(value, key=key, parent=parent):
                return result
        return ast.Constant(value=value)

    def _dict_to_ast(self, d, **kwargs):
        """Convert dictionary to AST node"""
        if not is_dict(d):
            return False

        keys = []
        values = []

        for k, v in d.items():
            # keys.append(ast.Constant(value=k))
            keys.append(k)
            values.append(self._value_to_ast(v, k, d))

        if not all(is_valid_variable_name(k) for k in keys):
            return ast.Dict(
                keys=[ast.Constant(value=k) for k in keys],
                values=values
            )
        else:
            # If possible, make dict beautiful
            return ast.Call(
                func=ast.Name(id='dict', ctx=ast.Load()),
                args=[],
                keywords=[ast.keyword(arg=k, value=v) for k, v in zip(keys, values)],
            )

    def _list_to_ast(self, lst, **kwargs):
        """Convert list to AST node"""
        if not (isinstance(lst, tuple) or is_list(lst)):
            return False

        return ast.List(
            elts=[self._value_to_ast(item) for item in lst],
            ctx=ast.Load()
        )

    def generate_config_function(self, obj, f_name='make_cfg'):
        """Generate config function, handle objects with extra attributes"""
        # Clear previously recorded objects
        if hasattr(self, '_objects_with_attrs'):
            delattr(self, '_objects_with_attrs')
        self._objects_with_attrs = {}

        # Create function body
        body = []

        # Create return value
        return_value = self._value_to_ast(obj)

        # Add return statement
        body.append(ast.Return(value=return_value))

        # Create function definition
        decorator = ast.Name(id='neko_cfg', ctx=ast.Load())
        func_def = ast.FunctionDef(
            name=f_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                kwarg=None,
                arg=None
            ),
            body=body,
            decorator_list=[decorator]
        )

        # Create module
        module = ast.Module(
            body=[func_def],
            type_ignores=[]
        )

        # Fix AST line numbers and other information
        ast.fix_missing_locations(module)

        return module

    def generate_code(self, cfg):
        """Generate Python code"""
        self.required_imports.clear()
        self.seen_types.clear()
        module_ast = self.generate_config_function(cfg)
        code = ast.unparse(module_ast)
        imports = self._generate_imports()

        full_code = imports + code
        new_code, _ = FormatCode(full_code, style_config={
            'based_on_style': 'facebook',
            'column_limit': 120  # Override facebook style line length
        })
        return new_code

    def _generate_imports(self):
        """Generate import statements"""
        import_lines = []

        # Direct imports
        if 'direct' in self.required_imports:
            for stmt in self.required_imports['direct']:
                if stmt.startswith('#'):
                    import_lines.append(stmt)  # Comment
                else:
                    import_lines.append(stmt)  # Normal import

        # Group imports by module
        for module, names in sorted(self.required_imports.items()):
            if module == 'direct':
                continue

            if module == 'torch' and ('dtype' in names or 'device' in names):
                import_lines.append("import torch")
                if 'device' in names:
                    names.remove('device')
                if 'dtype' in names:
                    names.remove('dtype')

            if '*' in names:
                import_lines.append(f"import {module}")
            elif module == 'pathlib' and 'Path' in names:
                import_lines.append("from pathlib import Path")
            elif names:
                names_str = ', '.join(sorted(name for name in names if name != '*'))
                if names_str:
                    import_lines.append(f"from {module} import {names_str}")

        return "\n".join(import_lines) + "\n\n" if import_lines else ""
