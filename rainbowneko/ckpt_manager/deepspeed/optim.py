import re
from itertools import chain

import torch
from deepspeed.checkpoint import (BASE_OPTIMIZER_STATE, GROUP_PADDINGS, OPTIMIZER_STATE_DICT, PARAM_GROUPS,
                                  PIPELINE_REPLICATED_PARAMETER_PATTERNS, TP_REPLICATED_PARAMETER_PATTERNS, PARAMETER_TO_AVERAGE_PATTERNS,
                                  PARAMETER_WITH_ROW_PARALLELISM_PATTERNS, VOCABULARY_PARAMETER_PATTERNS,
                                  PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0, PARAMETER_WITH_SUB_PARAMS, SubparamShape,
                                  UNIVERSAL_CHECKPOINT_VERSION_KEY, UNIVERSAL_CHECKPOINT_VERSION_VALUE, SINGLE_PARTITION_OF_FP32_GROUPS,
                                  PARAM_SLICE_MAPPINGS)

GROUP_STATE_KEY = 'state'


def _get_optimizer_state(sd, state_key):
    optimizer_state = sd.get(OPTIMIZER_STATE_DICT, None)
    if optimizer_state is None:
        return None

    return optimizer_state.get(state_key, None)


def _get_param_group_states(sd):
    optimizer_state = sd.get(OPTIMIZER_STATE_DICT, None)
    if optimizer_state is None:
        return None

    base_optimizer_state = optimizer_state.get(BASE_OPTIMIZER_STATE, None)
    if base_optimizer_state is None:
        return None

    return base_optimizer_state.get(GROUP_STATE_KEY, None)


def _strip_tensor_paddings(sd):
    param_group_states = _get_param_group_states(sd)
    if param_group_states is None:
        return

    group_paddings = _get_optimizer_state(sd, GROUP_PADDINGS)
    if group_paddings is None:
        return

    for key, group_state in param_group_states.items():
        if group_paddings[key] == 0:
            continue
        for state_name, state_value in group_state.items():
            if state_name != "step" and torch.is_tensor(state_value):
                raw_length = state_value.numel() - group_paddings[key]
                group_state[state_name] = torch.narrow(state_value, 0, 0, raw_length).clone()
            else:
                group_state[state_name] = state_value


def extract_zero_shards(optim_zero_sd, universal_checkpoint_info=None, pp_index=0):
    '''
    optimizer.state_dict():
    {
        'state': {
            0: {
                'step': int ,
                'exp_avg': tensor(...),
                'exp_avg_sq': tensor(...),
                ...
            },
            1: {'momentum_buffer': tensor(...), ...},
            ...
        },
        'param_groups': [
            {
                'lr': 0.01,
                'weight_decay': 0,
                ...
                'params': [0]
                'param_names' ['param0']  (optional)
            },
            {
                'lr': 0.001,
                'weight_decay': 0.5,
                ...
                'params': [1, 2, 3]'
                'param_names': ['param1', 'layer.weight', 'layer.bias'] (optional)
            }
        ]
    }
    '''
    param_slice_mappings = optim_zero_sd[PARAM_SLICE_MAPPINGS]
    pipeline_replicated_params = universal_checkpoint_info.get(PIPELINE_REPLICATED_PARAMETER_PATTERNS, [])
    # print(f'{pipeline_replicated_params=}')

    # dict
    state_groups = optim_zero_sd[BASE_OPTIMIZER_STATE]["state"]
    # list
    # fp32_groups = optim_zero_sd[SINGLE_PARTITION_OF_FP32_GROUPS]
    param_groups_cnt = len(state_groups)

    state = {}
    param_group_ids = {}

    for param_group_id in range(param_groups_cnt):
        flat_state = dict(
            exp_avg=state_groups[param_group_id]["exp_avg"].cpu(),
            exp_avg_sq=state_groups[param_group_id]["exp_avg_sq"].cpu(),
            # fp32=fp32_groups[param_group_id].cpu(),
        )

        if "step" in state_groups[param_group_id]:
            flat_state["step"] = state_groups[param_group_id]["step"].clone()

        for name, fragment_mapping in param_slice_mappings[param_group_id].items():
            if pp_index > 0 and any(re.match(pattern, name) for pattern in pipeline_replicated_params):
                # Skip tied weights that are replicated in first and last pp stages
                continue

            state[name] = {}
            for state_key in flat_state.keys():
                state_flat_tensor = flat_state[state_key]
                if state_key != "step" and torch.is_tensor(state_flat_tensor):
                    state_flat_tensor = state_flat_tensor.narrow(0, fragment_mapping.start, fragment_mapping.numel).clone()
                state[name][state_key] = state_flat_tensor
            param_group_ids[name] = param_group_id
    return state, param_group_ids


def _merge_zero_shards(slice_shards_list, slice_shape=None):
    '''
    :param slice_shards_list: [{
                'step': int ,
                'exp_avg': tensor(...),
                'exp_avg_sq': tensor(...),
                ...
            },
            {...}]
    '''
    slice_shards = {}
    for shard in slice_shards_list:
        for k, v in shard.items():
            if k not in slice_shards:
                slice_shards[k] = []
            slice_shards[k].append(v)

    slice_merged = {}
    for k, shards in slice_shards.items():
        if k == 'step':
            assert all(si.cpu() == shards[0].cpu() for si in shards), "All shards must have the same step value"
            slice = shards[0].clone()
        else:
            if slice_shape is None:
                slice = torch.cat(shards, dim=0)
            else:
                slice = torch.cat(shards, dim=0).reshape(slice_shape)
        slice_merged[k] = slice
    return slice_merged


def merge_tp_slices(sliced_zero_shards, universal_checkpoint_info, slice_shapes):
    replicated_parameters = universal_checkpoint_info.get(TP_REPLICATED_PARAMETER_PATTERNS, [])
    parameters_to_average = universal_checkpoint_info.get(PARAMETER_TO_AVERAGE_PATTERNS, [])
    parameters_with_row_parallelism = universal_checkpoint_info.get(PARAMETER_WITH_ROW_PARALLELISM_PATTERNS, [])
    vocabulary_parameters = universal_checkpoint_info.get(VOCABULARY_PARAMETER_PATTERNS, [])
    parameters_with_2_sub_params_cat_dim_0 = universal_checkpoint_info.get(PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0, [])
    parameter_with_sub_params = universal_checkpoint_info.get(PARAMETER_WITH_SUB_PARAMS, [])

    unmatched_patterns = set(replicated_parameters + parameters_to_average + parameters_with_row_parallelism +
                             vocabulary_parameters + parameters_with_2_sub_params_cat_dim_0)
    unmatched_patterns.update(chain.from_iterable(SubparamShape(**s).patterns for s in parameter_with_sub_params))

    def get_matched_pattern(patterns_, name_):
        matched_ = [pattern_ for pattern_ in patterns_ if re.match(pattern_, name_)]
        assert len(matched_) <= 1, f'Got more than one matching patterns={matched_} for {name_}'
        if matched_:
            pattern_ = matched_[0]
            unmatched_patterns.discard(pattern_)
            return pattern_
        return None

    def get_matched_sub_params_pattern(name_):
        for subparam_shape_dict in parameter_with_sub_params:
            subparam_shape = SubparamShape(**subparam_shape_dict)
            for pattern_ in subparam_shape.patterns:
                if re.match(pattern_, name_):
                    unmatched_patterns.discard(pattern_)
                    return subparam_shape
        return None

    full_states = {}
    for param_id, (name, zero_shards_list) in enumerate(sliced_zero_shards.items()):
        if len(zero_shards_list) == 0:
            # Freezed parameter
            continue
        elif isinstance(zero_shards_list[0], dict):  # dp only
            slice_merged = _merge_zero_shards(zero_shards_list, slice_shapes[name])
            full_states[param_id] = slice_merged
        else:  # with tp; zero_shards_list: [[{states},...],...]
            slices_merged = [_merge_zero_shards(zero_shards_list_tp, slice_shapes[name]) for zero_shards_list_tp in zero_shards_list]
            tp_degree = len(slices_merged)
            slice_tp_merged = {'step': slices_merged[0]['step']}

            matched_sub_params_shape = get_matched_sub_params_pattern(name)
            for state in ("fp32", "exp_avg", "exp_avg_sq"):
                if state not in slices_merged[0]:
                    continue
                slices = [slice_merged[state] for slice_merged in slices_merged]

                # print(f"Expected shape: {shape}")
                # print(f"Fragment sizes:", list(frag.shape for frag in slices))
                if get_matched_pattern(replicated_parameters, name):
                    if len(slices) > 1:
                        assert all([slices[0].equal(other_slice) for other_slice in slices[1:]])
                    param = slices[0]
                    # print(f'replicate {name} using first slice')
                elif get_matched_pattern(parameters_to_average, name):
                    param = sum(slices) / len(slices)
                    # print(f'merge {name} using average')
                elif get_matched_pattern(parameters_with_2_sub_params_cat_dim_0, name):
                    cat_dim = 0
                    chunked_slices = [torch.chunk(s, 2, dim=cat_dim) for s in slices]
                    merged_chunks_0 = torch.cat([s[0] for s in chunked_slices], dim=cat_dim)
                    merged_chunks_1 = torch.cat([s[1] for s in chunked_slices], dim=cat_dim)
                    param = torch.cat([merged_chunks_0, merged_chunks_1], dim=cat_dim)
                elif matched_sub_params_shape:
                    merged_chunks = []
                    partition_dim = matched_sub_params_shape.partition_dim

                    sub_dim_sizes = matched_sub_params_shape.shape[partition_dim]
                    if not isinstance(sub_dim_sizes, tuple):
                        sub_dim_sizes = (sub_dim_sizes,)

                    partition_shape = [sum(d) if isinstance(d, tuple) else d for d in matched_sub_params_shape.shape]
                    partition_shape = [d // tp_degree if i == partition_dim else d for i, d in enumerate(partition_shape)]
                    slices = [s.view(partition_shape) for s in slices]

                    offset = 0
                    for sub_dim_size in sub_dim_sizes:
                        part_sub_dim_size = sub_dim_size // tp_degree
                        merged_chunks.append(
                            torch.cat([s.narrow(partition_dim, offset, part_sub_dim_size) for s in slices], dim=partition_dim))
                        offset += part_sub_dim_size
                    param = torch.cat(merged_chunks, dim=partition_dim)
                else:
                    cat_dim = 1 if get_matched_pattern(parameters_with_row_parallelism, name) else 0
                    # print(f"merge {name} with CAT DIM: {cat_dim}")
                    param = torch.cat(slices, dim=cat_dim)

                if get_matched_pattern(vocabulary_parameters, name):
                    # print(f"Before {param.shape=}")
                    # strip padding
                    original_vocab_size = universal_checkpoint_info['original_vocab_size']
                    param = param[:original_vocab_size, :]
                    # print(f"After {param.shape=}")

                # print(f"Final shape: {param.shape}")
                slice_tp_merged[state] = param
            full_states[param_id] = slice_tp_merged

    return full_states

def _inject_missing_state():
    universal_checkpoint_info = {UNIVERSAL_CHECKPOINT_VERSION_KEY: UNIVERSAL_CHECKPOINT_VERSION_VALUE}
    return universal_checkpoint_info

def zero_optimizer_state_to_torch(zero_state_list, names, param_shapes_list, universal_checkpoint_info=None):
    # 0. Preprocess ZeRO states
    for sd in zero_state_list:
        _strip_tensor_paddings(sd)

    if not universal_checkpoint_info:
        universal_checkpoint_info = _inject_missing_state()

    # 1. Extracting ZeRO fragments
    sliced_zero_shards = {name:[] for name in names}
    param_group_ids_list = []
    for zero_state in zero_state_list:
        sliced_states, param_group_ids = extract_zero_shards(zero_state, universal_checkpoint_info)
        for name, sd in sliced_states.items():
            sliced_zero_shards[name].append(sd)
        param_group_ids_list.append(param_group_ids)

    # 2 Find param group for each slices
    param_groups = zero_state_list[0][BASE_OPTIMIZER_STATE][PARAM_GROUPS]
    for pg in param_groups:
        pg['params'] = []
    for param_id, name in enumerate(sliced_zero_shards.keys()):
        for param_group_ids in param_group_ids_list:
            if name in param_group_ids:
                pg_id = param_group_ids[name]
                param_groups[pg_id]['params'].append(param_id)
                break

    # 3.1 Get all param shapes
    slice_shapes = sum(param_shapes_list, [])
    # fix back to normal flat dict, merge duplicates for tp>1
    slice_shapes = {k: v for d in slice_shapes for k, v in d.items()}

    # 3.2 Merge all ZeRO fragments
    full_states = merge_tp_slices(sliced_zero_shards, universal_checkpoint_info, slice_shapes)

    state_dict = {'state': full_states, 'param_groups': param_groups}
    return state_dict


def extract_zero_slices(param_slice_mappings_list, pp_index=0):
    slices_shards = {}
    for dp_index, param_slice_mappings in enumerate(param_slice_mappings_list):
        param_groups_cnt = len(param_slice_mappings)
        for param_group_id in range(param_groups_cnt):
            for name, fragment_mapping in param_slice_mappings[param_group_id].items():
                if pp_index > 0:
                    # Skip tied weights that are replicated in first and last pp stages
                    continue

                if name not in slices_shards:
                    slices_shards[name] = {}
                    offset = 0
                else:
                    last_dp = max(slices_shards[name])
                    _, pre_numel, pre_offset = slices_shards[name][last_dp]
                    offset = pre_offset + pre_numel

                slices_shards[name][dp_index] = (fragment_mapping.start, fragment_mapping.numel, offset)
    return slices_shards

def load_torch_optimizer_to_zero(optimizer, optim_sd, param_slice_mappings_list, names, dp_index=0):
    zero_sd = optimizer.state_dict()
    slices_shards = extract_zero_slices(param_slice_mappings_list)
    fp32_groups = zero_sd[SINGLE_PARTITION_OF_FP32_GROUPS]
    group_paddings = zero_sd[GROUP_PADDINGS]
    optim_sd["state"] = {int(k):v for k, v in optim_sd["state"].items()}

    name2pg = {name:i for i, name in enumerate(names) if name in slices_shards}
    for param_group_id, param_slice_mapping in enumerate(param_slice_mappings_list[dp_index]):
        param = fp32_groups[param_group_id]
        zero_sd[BASE_OPTIMIZER_STATE]["state"][param_group_id] = {
            'exp_avg': torch.zeros(len(param)+group_paddings[param_group_id], dtype=torch.float32, device=param.device),
            'exp_avg_sq': torch.zeros(len(param)+group_paddings[param_group_id], dtype=torch.float32, device=param.device),
        }

        for name, fragment_mapping in param_slice_mapping.items():
            zero_sd[BASE_OPTIMIZER_STATE]["state"][param_group_id]["step"] = optim_sd["state"][name2pg[name]]["step"].to(dtype=torch.float32, device=param.device)
            for state in ('exp_avg', 'exp_avg_sq'):
                flat_state = zero_sd[BASE_OPTIMIZER_STATE]["state"][param_group_id][state]
                start, numel, offset = slices_shards[name][dp_index]
                flat_state[start:start + numel] = optim_sd["state"][name2pg[name]][state].flatten()[offset:offset + numel].to(dtype=torch.float32, device=param.device)

    optimizer.load_state_dict({dp_index: zero_sd})
