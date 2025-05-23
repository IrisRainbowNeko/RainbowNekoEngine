import ast
import inspect
import types
from collections import defaultdict
from pathlib import Path

from yapf.yapflib.yapf_api import FormatCode

from rainbowneko.utils import is_valid_variable_name, is_dict, is_list


class LambdaFinder(ast.NodeVisitor):
    def __init__(self):
        self.lambda_list = []

    def visit_Lambda(self, node):
        self.lambda_list.append(node)
        self.generic_visit(node)


class ConfigCodeReconstructor:
    def __init__(self):
        self.type_handlers = {}
        self.required_imports = defaultdict(set)
        self.seen_types = set()

        # Register basic handlers
        self.register_type_handler('Path', self._handle_path)
        self.register_type_handler('_custom_object', self._handle_custom_object)
        self.register_type_handler('_object_instance', self._handle_object_instance)
        self.register_type_handler('_module', self._handle_module)
        self.register_type_handler('_lambda', self._handle_lambda)

    def register_type_handler(self, type_name, handler_func):
        """Register type handler"""
        self.type_handlers[type_name] = handler_func

    def _detect_type(self, obj):
        """Detect object type and automatically identify and add import requirements"""
        self.required_imports['rainbowneko.parser'].add('neko_cfg')

        # Path object detection
        if isinstance(obj, Path):
            self.required_imports['pathlib'].add('Path')
            return 'Path', self.type_handlers.get('Path')

        if isinstance(obj, types.ModuleType):
            parts = obj.__name__.split('.')
            module_path = '.'.join(parts[:-1])
            module_name = parts[-1]
            self.required_imports[module_path].add(module_name)
            return '_module', self.type_handlers.get('_module')

        # Detect lambda expressions
        if callable(obj) and isinstance(obj, types.FunctionType) and obj.__name__ == '<lambda>':
            return '_lambda', self.type_handlers.get('_lambda')

        # Dictionary-form custom objects
        if is_dict(obj) and '_target_' in obj:
            type_name = obj.get('_target_')

            # Try to automatically add type imports
            if isinstance(type_name, str):
                if '.' in type_name:  # If type name contains module path
                    parts = type_name.split('.')
                    module_path = '.'.join(parts[:-1])
                    class_name = parts[-1]
                    self.required_imports[module_path].add(class_name)
            elif callable(type_name):
                # Add import requirements
                if hasattr(type_name, '__module__') and type_name.__module__ != '__main__':
                    module_name = type_name.__module__
                    if hasattr(type_name, '__qualname__'):  # for staticmethod and classmethod
                        name = type_name.__qualname__.split('.')[0]
                    elif hasattr(type_name, '__name__'):
                        name = type_name.__name__
                    self.required_imports[module_name].add(name)

            return '_custom_object', self.type_handlers.get('_custom_object')

        # Instantiated object detection
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
                    if '.' in module_name:
                        specific_module = module_name
                        self.required_imports[specific_module].add(class_name)
                    else:
                        self.required_imports[module_name].add(class_name)

            return '_object_instance', self.type_handlers.get('_object_instance')

        return None, None

    def _handle_path(self, obj, **kwargs):
        """Handle Path objects"""
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
        type_name = obj.get('_target_', 'CustomObject')
        obj_data = {k: v for k, v in obj.items() if k != '_target_'}

        if callable(type_name):
            if isinstance(type_name, (types.FunctionType, types.MethodType, type)):
                # Function or method
                if hasattr(type_name, '__qualname__'):  # for staticmethod and classmethod
                    class_name = type_name.__qualname__
                else:
                    class_name = type_name.__name__
            else:
                # Other callable objects
                self.required_imports['direct'].add(f"# Cannot directly reference callable object: {type_name}")
                return ast.Name(id=f"__CALLABLE_{type_name}__", ctx=ast.Load())
        else:
            # Extract class name (handle possible module path)
            if '.' in type_name:
                class_name = type_name.split('.')[-1]
            else:
                class_name = type_name

        # Create custom class call
        args = []
        keywords = []

        for k, v in obj_data.items():
            if k == '_args_':
                for arg in v:
                    args.append(self._value_to_ast(arg))
            else:
                keywords.append(ast.keyword(arg=k, value=self._value_to_ast(v, k)))

        return ast.Call(
            func=ast.Name(id=class_name, ctx=ast.Load()),
            args=args,
            keywords=keywords
        )

    def _handle_object_instance(self, obj, **kwargs):
        """Handle instantiated objects, treating non-constructor parameters as separate assignment statements"""
        class_name = obj.__class__.__name__
        module_name = obj.__class__.__module__

        try:
            # Check if it's torch.dtype
            if class_name == 'dtype' and module_name == 'torch':
                module_name, class_name = str(obj).rsplit('.', maxsplit=1)
                return ast.Attribute(value=ast.Name(id=module_name, ctx=ast.Load()), attr=class_name, ctx=ast.Load())

            if class_name == 'device' and module_name == 'torch':
                return ast.Call(
                    func=ast.Attribute(value=ast.Name(id=module_name, ctx=ast.Load()), attr=class_name, ctx=ast.Load()),
                    args=[
                        ast.Constant(value=obj.type),
                        ast.Constant(value=obj.index),
                    ],
                    keywords=[]
                )

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
                all_attrs.update({k: v for k, v in obj.__dict__.items() if not k.startswith('_')})

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
        value = module.__name__.split('.')[-1]
        return ast.Name(id=value, ctx=ast.Load())

    def _handle_lambda(self, lambda_obj, **kwargs):
        """Handle lambda expressions"""
        try:
            # Check if source code can be obtained
            source = inspect.getsource(lambda_obj).strip()
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

    def _value_to_ast(self, value, key=None, parent=None):
        """Convert value to AST node"""
        # Detect type
        type_name, handler = self._detect_type(value)

        if handler:
            return handler(value, key=key, parent=parent)
        elif is_dict(value):
            return self._dict_to_ast(value)
        elif isinstance(value, tuple) or is_list(value):
            return ast.List(
                elts=[self._value_to_ast(item) for item in value],
                ctx=ast.Load()
            )
        else:
            return ast.Constant(value=value)

    def _dict_to_ast(self, d):
        """Convert dictionary to AST node"""
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
