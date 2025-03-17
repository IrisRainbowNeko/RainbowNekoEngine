import ast
import importlib
import inspect
import os
import re
import shutil
from typing import Union
from types import ModuleType

from omegaconf import OmegaConf

from .yaml_cfg import YamlCfgParser

def get_rel_path(path):
    if not isinstance(path, str):
        path = path.__file__
    current_dir = os.getcwd()
    return os.path.relpath(path, current_dir)

class CallTransformer(ast.NodeTransformer):
    transform_parent = (ast.Call, ast.Expr, ast.Dict, ast.List)
    node_skip = ast.Lambda

    def __init__(self):
        self.parent_stack = []

    def visit(self, node):
        if isinstance(node, self.node_skip):
            return node

        # 在遍历到每个节点时，将当前节点推入栈
        self.parent_stack.append(node)
        result = super().visit(node)
        # 完成当前节点的处理后，将其从栈中弹出
        self.parent_stack.pop()
        return result

    def find_call(self, node: ast.Attribute):
        ''' A().b.c() => [A(), b, c()] '''
        node_list = []
        while isinstance(node, ast.Attribute):
            node_list.append(node)
            node = node.value

        if isinstance(node, ast.Call):
            node_list.append(node)
            return node_list[::-1]
        else:
            return None

    def visit_Call(self, node):
        # 创建一个新的Call节点，调用dict函数并传递关键字参数
        call_node = ast.Call(
            func=ast.Name(id='dict', ctx=ast.Load()),
            args=[],
            keywords=[]
        )

        partial_flag = False

        # skip node that cannot transform
        parent_node = self.parent_stack[-2] if len(self.parent_stack) > 1 else None
        if not isinstance(parent_node, self.transform_parent):
            return node

        if isinstance(node.func, (ast.Attribute, ast.Call)):
            node_list = self.find_call(node.func)
            if node_list is None:
                call_node.keywords.append(
                    ast.keyword(arg='_target_', value=ast.Attribute(attr=node.func.attr, value=node.func.value)))
            else:
                prev_node = self.visit_Call(node_list[0])
                for node_attr in node_list[1:]:
                    prev_node = ast.Call(
                        func=ast.Name(id='dict', ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(arg='_target_', value=ast.Name(id='getattr', ctx=ast.Load())),
                            ast.keyword(arg='_args_', value=ast.List(elts=[prev_node, ast.Constant(value=node_attr.attr)], ctx=ast.Load())),
                        ]
                    )
                call_node.keywords.append(
                    ast.keyword(arg='_target_', value=prev_node)
                )

        else:
            if node.func.id == 'partial':
                partial_flag = True
            elif not node.func.id == 'dict':
                call_node.keywords.append(ast.keyword(arg='_target_', value=ast.Name(id=node.func.id)))


        # 处理位置参数
        if node.args:
            args_list = [self.visit(arg) for arg in node.args]
            if partial_flag:
                call_node.keywords.append(ast.keyword(arg='_target_', value=args_list[0]))
                call_node.keywords.append(ast.keyword(arg='_partial_', value=ast.NameConstant(value=True)))
                if len(args_list) > 1:
                    args_list = ast.List(elts=args_list[1:], ctx=ast.Load())
                    call_node.keywords.append(ast.keyword(arg='_args_', value=args_list))
            else:
                # 创建一个列表节点，包含所有的位置参数
                args_list = ast.List(elts=args_list, ctx=ast.Load())
                # 将列表节点作为字典的值
                call_node.keywords.append(ast.keyword(arg='_args_', value=args_list))

        # 处理关键字参数
        if node.keywords:
            for keyword in node.keywords:
                call_node.keywords.append(ast.keyword(arg=keyword.arg, value=self.visit(keyword.value)))

        # 替换原始的Call节点
        return call_node


class PythonCfgParser(YamlCfgParser):
    def __init__(self):
        super().__init__()
        self.cfg_dict = {}

    def get_code(self, func):
        # 获取函数源代码
        source_code = inspect.getsource(func)

        # 分离函数定义和函数体
        start_index = re.search(r':\s*\n', source_code).end()  # 找到第一个换行符后的索引
        function_body = source_code[start_index:].strip()
        return function_body

    def get_default_kwargs(self, func):
        sig = inspect.signature(func)
        return {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}

    def transform_code(self, code):
        # 解析代码为AST
        tree = ast.parse(code)
        # 应用转换器
        transformer = CallTransformer()
        new_tree = transformer.visit(tree)

        # 将AST转换回代码字符串
        new_code = ast.unparse(new_tree)

        # self.print_code(new_code)

        return new_code

    def print_code(self, code):
        # 使用yapf格式化代码
        from yapf.yapflib.yapf_api import FormatCode
        new_code, _ = FormatCode(code, style_config='facebook')
        print(new_code)

    def resolve_sub_cfgs(self, module, cfg):
        if isinstance(cfg, dict):
            if '_target_' in cfg and getattr(cfg['_target_'], '_neko_cfg_', False):
                code = self.get_code(cfg['_target_'])
                code_format = self.transform_code(code)
                kwargs = self.get_default_kwargs(cfg['_target_'])
                del cfg['_target_']
                cfg = eval(code_format, vars(module), {**kwargs, **cfg})
                return cfg

            for key, value in cfg.items():
                if isinstance(value, dict):
                    res = self.resolve_sub_cfgs(module, value)
                    if res is not None:
                        cfg[key] = res
                if isinstance(value, list):
                    self.resolve_sub_cfgs(module, value)
        elif isinstance(cfg, list):
            for idx, value in enumerate(cfg):
                if isinstance(value, dict):
                    res = self.resolve_sub_cfgs(module, value)
                    if res is not None:
                        cfg[idx] = res
                elif isinstance(value, list):
                    self.resolve_sub_cfgs(module, value)

    def load_cfg(self, path: Union[str, ModuleType], trans=True):
        if isinstance(path, str):
            # record for save
            if len(self.cfg_dict) == 0:
                self.cfg_dict['cfg.py'] = path
            else:
                self.cfg_dict[path] = path

            # load module
            module_name = os.path.splitext(os.path.basename(path))[0]
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        elif isinstance(path, ModuleType):
            # record for save
            cfg_path = get_rel_path(path)
            if len(self.cfg_dict) == 0:
                self.cfg_dict['cfg.py'] = cfg_path
            else:
                self.cfg_dict[cfg_path] = cfg_path

            module = path
        else:
            raise ValueError(f'base must be str or module. But type of base is {type(path)}')

        if trans:
            code = self.get_code(module.make_cfg)
            code_format = self.transform_code(code)
            cfg = eval(code_format, vars(module))
        else:
            cfg = module.config
        self.resolve_sub_cfgs(module, cfg)

        return OmegaConf.create(cfg, flags={"allow_objects": True})

    def save_configs(self, cfg, path, name='cfg'):
        for dst, src in self.cfg_dict.items():
            if dst == 'cfg.py':
                path_dst = os.path.join(path, f'{name}.py')
            else:
                path_dst = os.path.join(path, dst)
            os.makedirs(os.path.dirname(path_dst), exist_ok=True)
            shutil.copy2(src, path_dst)
