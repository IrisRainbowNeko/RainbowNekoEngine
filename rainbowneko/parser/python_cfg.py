import ast
import importlib
import inspect
import linecache
import os
import random
import re
import shutil
import string
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Any

from omegaconf import OmegaConf
from yapf.yapflib.yapf_api import FormatCode

from .cfg2py import ConfigCodeReconstructor
from .yaml_cfg import YamlCfgParser


def get_rel_path(path):
    if not isinstance(path, str):
        path = path.__file__
    current_dir = os.getcwd()
    return os.path.relpath(path, current_dir)


def random_filename(length=8, suffix=".py"):
    chars = string.ascii_letters + string.digits  # 字母 + 数字
    random_part = ''.join(random.choices(chars, k=length))
    return f"{random_part}{suffix}"


disable_neko_cfg = nullcontext()


class CallTransformer(ast.NodeTransformer):
    transform_parent = (ast.Call, ast.Expr, ast.Dict, ast.List, ast.Return, ast.Assign)
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
                    ast.keyword(arg='_target_', value=ast.Attribute(attr=node.func.attr, value=node.func.value, ctx=ast.Load())))
            else:
                prev_node = self.visit_Call(node_list[0])
                for node_attr in node_list[1:]:
                    prev_node = ast.Call(
                        func=ast.Name(id='dict', ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(arg='_target_', value=ast.Name(id='getattr', ctx=ast.Load())),
                            ast.keyword(arg='_args_',
                                        value=ast.List(elts=[prev_node, ast.Constant(value=node_attr.attr)], ctx=ast.Load())),
                        ]
                    )
                call_node.keywords.append(
                    ast.keyword(arg='_target_', value=prev_node)
                )

        else:
            if node.func.id == 'partial':
                partial_flag = True
            elif node.func.id != 'dict':
                call_node.keywords.append(ast.keyword(arg='_target_', value=ast.Name(id=node.func.id, ctx=ast.Load())))

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
            for i, keyword in enumerate(node.keywords):
                if keyword.arg is None and isinstance(keyword.value, ast.Call):  # merge '**neko_cfg()' to dict
                    call_node.keywords.append(ast.keyword(arg=f'_merge_{i}_', value=self.visit(keyword.value)))
                else:
                    call_node.keywords.append(ast.keyword(arg=keyword.arg, value=self.visit(keyword.value)))

        # 替换原始的Call节点
        return call_node

    def visit_With(self, node):
        for item in node.items:
            if isinstance(item.context_expr, ast.Name) and item.context_expr.id == 'disable_neko_cfg':
                return node
        super().generic_visit(node)
        return node


class PythonCfgParser(YamlCfgParser):
    def __init__(self):
        super().__init__()
        self.cfg_dict = {}

    def print_code(self, code):
        # format code with yapf
        new_code, _ = FormatCode(code, style_config='facebook')
        print(new_code)

    def compile_cfg(self, func):
        source = inspect.getsource(func)  # get source code
        tree = ast.parse(source)  # AST tree
        tree_func = tree.body[0]
        tree_func.decorator_list = [x for x in tree_func.decorator_list if x.id != 'neko_cfg']

        modified_tree = CallTransformer().visit(tree)
        ast.fix_missing_locations(modified_tree)

        # add source code for new function
        filename = random_filename()
        new_code = ast.unparse(modified_tree)
        # for easily detect lambda in ConfigCodeReconstructor
        new_code, _ = FormatCode(new_code, style_config={
            'based_on_style': 'facebook',
            'column_limit': 100  # 覆盖 facebook 样式的行长度
        })
        lines = [line + '\n' for line in new_code.splitlines()]
        linecache.cache[filename] = (len(source), None, lines, filename)
        modified_tree = ast.parse(new_code)  # rebuild lineno

        # compile modified code
        code = compile(modified_tree, filename=filename, mode="exec")
        namespace = func.__globals__
        exec(code, namespace)
        f_new = namespace[func.__name__]
        f_new._neko_cfg_ = True
        return f_new

    def resolve_sub_cfgs(self, cfg):
        if isinstance(cfg, dict):
            if '_target_' in cfg and getattr(cfg['_target_'], '_neko_cfg_', False):
                target = cfg.pop('_target_')
                if '_args_' in cfg:
                    args = cfg.pop('_args_')
                    cfg = target(*args, **cfg)
                else:
                    cfg = target(**cfg)
                res = self.resolve_sub_cfgs(cfg)
                return res or cfg

            cfg_new = {}
            for key, value in cfg.items():
                if isinstance(value, dict):
                    if re.match(r'_merge_\d+_', key):
                        res = self.resolve_sub_cfgs(value)
                        cfg_new.update(**res)
                    else:
                        res = self.resolve_sub_cfgs(value)
                        cfg_new[key] = res or value
                elif isinstance(value, list):
                    self.resolve_sub_cfgs(value)
                    cfg_new[key] = value
                else:
                    cfg_new[key] = value
            return cfg_new
        elif isinstance(cfg, list):
            for idx, value in enumerate(cfg):
                if isinstance(value, dict):
                    res = self.resolve_sub_cfgs(value)
                    if res is not None:
                        cfg[idx] = res
                elif isinstance(value, list):
                    self.resolve_sub_cfgs(value)

    def load_cfg(self, path: str | ModuleType):
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

        if getattr(module.make_cfg, '_neko_cfg_', False):
            cfg = module.make_cfg()
        else:
            f_cfg = self.compile_cfg(module.make_cfg)
            cfg = f_cfg()
        cfg = self.resolve_sub_cfgs(cfg) or cfg

        return OmegaConf.create(cfg, flags={"allow_objects": True})

    def save_configs(self, cfg: Dict, path: str | Path, name='full_cfg'):
        path = Path(path)
        for dst, src in self.cfg_dict.items():
            path_dst = path / dst
            os.makedirs(os.path.dirname(path_dst), exist_ok=True)
            shutil.copy2(src, path_dst)

        try:
            coder = ConfigCodeReconstructor()
            cfg_code = coder.generate_code(cfg)
            with open(path / f'{name}.py', 'w') as f:
                f.write(cfg_code)
        except:
            print('Reconstruct code from config failed.')
