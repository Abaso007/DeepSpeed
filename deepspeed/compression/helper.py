# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from .basic_layer import Embedding_Compress, LinearLayer_Compress, Conv2dLayer_Compress, BNLayer_Compress, ColumnParallelLinear_Compress, RowParallelLinear_Compress
from .constants import *


def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)


def module_replacement(model, module_name, compression_technique=None, mpu=None):
    """
    Replace a module with a new module.
    Args:
        model (`torch.nn.Module`)
            The model to replace the module in.
        module_name (`str`)
            The name of the module to replace.
        compression_technique (`str`)
            The compression technique to use for the new module.
    """

    # Get the old module
    old_module = recursive_getattr(model, module_name)

    need_bias = bool(hasattr(old_module, 'bias') and old_module.bias is not None)
    # Initialize the new module
    if isinstance(old_module, LinearLayer_Compress):
        new_module = old_module
    elif isinstance(old_module, torch.nn.Linear):
        new_module = LinearLayer_Compress(old_module.in_features, old_module.out_features,
                                          bias=need_bias).to(device=old_module.weight.device,
                                                             dtype=old_module.weight.dtype)
        new_module.weight.data = old_module.weight.data
        if need_bias:
            new_module.bias.data = old_module.bias.data
    elif isinstance(old_module, (Conv2dLayer_Compress, torch.nn.Conv2d)):
        if isinstance(old_module, Conv2dLayer_Compress):
            new_module = old_module
        else:
            new_module = Conv2dLayer_Compress(old_module.in_channels, old_module.out_channels, old_module.kernel_size, old_module.stride, old_module.padding, \
                                            old_module.dilation, old_module.groups, need_bias, \
                                            old_module.padding_mode).to(device=old_module.weight.device, dtype=old_module.weight.dtype)
            new_module.weight.data = old_module.weight.data
            if need_bias:
                new_module.bias.data = old_module.bias.data
    elif isinstance(old_module, torch.nn.BatchNorm2d):
        new_module = BNLayer_Compress(old_module.num_features, old_module.eps, old_module.momentum, old_module.affine,
                                      old_module.track_running_stats).to(old_module.weight.device,
                                                                         old_module.weight.dtype)
        new_module.weight.data = old_module.weight.data
        if need_bias:
            new_module.bias.data = old_module.bias.data
        new_module.running_mean.data = old_module.running_mean.data
        new_module.running_var.data = old_module.running_var.data
    elif isinstance(old_module, (Embedding_Compress, torch.nn.Embedding)):
        if isinstance(old_module, Embedding_Compress):
            new_module = old_module
        else:
            new_module = Embedding_Compress(old_module.num_embeddings, old_module.embedding_dim, old_module.padding_idx, old_module.max_norm, old_module.norm_type, \
                                        old_module.scale_grad_by_freq, old_module.sparse).to(device=old_module.weight.device, dtype=old_module.weight.dtype)
            new_module.weight.data = old_module.weight.data
    elif mpu is not None and (
        isinstance(
            old_module,
            (ColumnParallelLinear_Compress, mpu.ColumnParallelLinear),
        )
    ):
        if isinstance(old_module, ColumnParallelLinear_Compress):
            new_module = old_module
        else:
            new_module = ColumnParallelLinear_Compress(mpu,
                                                       old_module.input_size,
                                                       old_module.output_size,
                                                       gather_output=old_module.gather_output,
                                                       skip_bias_add=old_module.skip_bias_add,
                                                       bias=need_bias).to(device=old_module.weight.device,
                                                                          dtype=old_module.weight.dtype)
            new_module.weight.data = old_module.weight.data
            if need_bias:
                new_module.bias.data = old_module.bias.data
    elif mpu is not None and (
        isinstance(
            old_module, (RowParallelLinear_Compress, mpu.RowParallelLinear)
        )
    ):
        if isinstance(old_module, RowParallelLinear_Compress):
            new_module = old_module
        else:
            new_module = RowParallelLinear_Compress(mpu,
                                                    old_module.input_size,
                                                    old_module.output_size,
                                                    input_is_parallel=old_module.input_is_parallel,
                                                    skip_bias_add=old_module.skip_bias_add,
                                                    bias=need_bias).to(device=old_module.weight.device,
                                                                       dtype=old_module.weight.dtype)
            new_module.weight.data = old_module.weight.data
            if need_bias:
                new_module.bias.data = old_module.bias.data
    else:
        new_module = None

    if compression_technique is not None:
        for k, v in compression_technique.items():
            if k == SPARSE_PRUNING:
                if v[SPARSE_PRUNING_ENABLED]:
                    new_module.enable_sparse_pruning(v[SPARSE_PRUNING_DENSE_RATIO], v[SPARSE_PRUNING_METHOD])
            elif k == ROW_PRUNING:
                if v[ROW_PRUNING_ENABLED]:
                    new_module.enable_row_pruning(v[ROW_PRUNING_DENSE_RATIO], v[ROW_PRUNING_METHOD])
            elif k == HEAD_PRUNING:
                if v[HEAD_PRUNING_ENABLED]:
                    new_module.enable_head_pruning(v[HEAD_PRUNING_DENSE_RATIO], v[HEAD_PRUNING_METHOD],
                                                   v[HEAD_PRUNING_NUM_HEADS])
            elif k == ACTIVATION_QUANTIZATION:
                if v[ACTIVATION_QUANTIZATION_ENABLED]:
                    new_module.enable_activation_quantization(v[ACTIVATION_QUANTIZE_BITS], v[ACTIVATION_QUANTIZE_TYPE],
                                                              v[ACTIVATION_QUANTIZE_RANGE])
            elif k == WEIGHT_QUANTIZATION:
                if v[WEIGHT_QUANTIZE_ENABLED]:
                    new_module.enable_weight_quantization(v[WEIGHT_QUANTIZE_START_BITS],
                                                          v[WEIGHT_QUANTIZE_TARGET_BITS],
                                                          v[WEIGHT_QUANTIZATION_PERIOD],
                                                          v[WEIGHT_QUANTIZE_IN_FORWARD_ENABLED],
                                                          v[WEIGHT_QUANTIZE_TYPE], v[WEIGHT_QUANTIZE_GROUPS])
            elif k == CHANNEL_PRUNING:
                if v[CHANNEL_PRUNING_ENABLED]:
                    new_module.enable_channel_pruning(v[CHANNEL_PRUNING_DENSE_RATIO], v[CHANNEL_PRUNING_METHOD])
            else:
                raise NotImplementedError(f'Compression technique {k} is not implemented')

    # Replace the old module with the new one
    recursive_setattr(model, module_name, new_module)


def is_module_compressible(module, mpu=None):
    ret = isinstance(
        module,
        (
            torch.nn.Linear,
            torch.nn.Conv2d,
            torch.nn.Embedding,
            torch.nn.BatchNorm2d,
        ),
    )

    if mpu is not None:
        ret = ret or isinstance(
            module, (mpu.RowParallelLinear, mpu.ColumnParallelLinear)
        )

    return ret


def compression_preparation(model, compression_techinique_list, mpu):
    """
    Prepare the compression techniques of a model.
    Args:
        model (`torch.nn.Module`)
            The model to prepare the compression techniques of.
        compression_techinique_list (`list`)
            The list of compression techniques to prepare the model to.
            list[]
    """
    # Here we first replace all module with our linear wrapper
    for module_name, module in model.named_modules():
        if is_module_compressible(module, mpu):
            module_replacement(model, module_name, mpu=mpu)
    for module_name_lists, _, compression_technique in compression_techinique_list:
        for mnl in module_name_lists:
            for module_name in mnl:
                module_replacement(model, module_name, compression_technique)

    return model


def fix_compression(model, module_name, compression_technique, mask=None, dim_reduction=False):
    """
    Fix the compression technique of a module.
    Args:
        model (`torch.nn.Module`)
            The model to fix the compression technique of.
        module_name (`str`)
            The name of the module to fix the compression technique of.
        compression_technique (`str`)
            The compression technique to fix the module to.
    """
    # Here we can make things much simpler by just replacing the module
    module = recursive_getattr(model, module_name)
    for k, v in compression_technique.items():
        if k == WEIGHT_QUANTIZATION and v[WEIGHT_QUANTIZE_IN_FORWARD_ENABLED] and v[WEIGHT_QUANTIZE_ENABLED]:
            return module.fix_weight_quantization()
        elif k == SPARSE_PRUNING and v[SPARSE_PRUNING_ENABLED]:
            return module.fix_sparse_pruning_helper()
        elif k == ROW_PRUNING and (v[ROW_PRUNING_ENABLED] or mask is not None):
            return module.fix_row_col_pruning_helper(mask, dim_reduction=dim_reduction)
        elif k == HEAD_PRUNING and (v[HEAD_PRUNING_ENABLED] or mask is not None):
            return module.fix_head_pruning_helper(mask, v[HEAD_PRUNING_NUM_HEADS], dim_reduction=dim_reduction)
        elif k == CHANNEL_PRUNING and (v[CHANNEL_PRUNING_ENABLED] or mask is not None):
            return module.fix_channel_pruning_helper(mask, dim_reduction=dim_reduction)


def convert_conv1d_to_linear(model, convert_type):
    '''
    This is a help function to convert conv1d to linear (e.g., convert GPT2 from HF)
    '''
    c_model = model.module if hasattr(model, 'module') else model
    for name, module in c_model.named_modules():
        if isinstance(module, convert_type):
            old_module = recursive_getattr(c_model, name)
            new_module = torch.nn.Linear(
                old_module.weight.data.size(0),
                old_module.weight.data.size(1),
                bias=old_module.bias is not None,
            )
            new_module.weight.data = old_module.weight.data.t().contiguous()
            if new_module.bias is not None:
                new_module.bias.data = old_module.bias.data.view(-1)

            recursive_setattr(c_model, name, new_module)

    return model
