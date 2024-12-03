import re
from typing import Dict, List, Union, Any

from torch import nn


def get_class_match_layer(class_name, block: nn.Module):
    if type(block).__name__ == class_name:
        return [""]
    else:
        return ["." + name for name, layer in block.named_modules() if type(layer).__name__ == class_name]


def get_match_layers(layers, all_layers, return_metas=False) -> Union[List[str], List[Dict[str, Any]]]:
    res = []
    for name in layers:
        metas = name.split(":")

        use_re = False
        pre_hook = False
        cls_filter = None
        for meta in metas[:-1]:
            if meta == "re":
                use_re = True
            elif meta == "pre_hook":
                pre_hook = True
            elif meta.startswith("cls("):
                cls_filter = meta[4:-1]

        name = metas[-1]
        if use_re:
            pattern = re.compile(name)
            match_layers = filter(lambda x: pattern.match(x) != None, all_layers.keys())
        else:
            match_layers = [name]

        if cls_filter is not None:
            match_layers_new = []
            for layer in match_layers:
                match_layers_new.extend([layer + x for x in get_class_match_layer(name[1], all_layers[layer])])
            match_layers = match_layers_new

        for layer in match_layers:
            if return_metas:
                res.append({"layer": layer, "pre_hook": pre_hook})
            else:
                res.append(layer)

    # Remove duplicates and keep the original order
    if return_metas:
        layer_set = set()
        res_unique = []
        for item in res:
            if item["layer"] not in layer_set:
                layer_set.add(item["layer"])
                res_unique.append(item)
        return res_unique
    else:
        return sorted(set(res), key=res.index)
