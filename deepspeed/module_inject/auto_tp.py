# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Automatic Tensor Parallelism
import re

from torch import nn
from .replace_policy import replace_policies


class AutoTP():

    def in_module_list(self, module_list):
        return any(type(item).__name__ == type(self).__name__ for item in module_list)

    def get_module_list(self):
        mlist = []
        for child in self.children():
            if isinstance(child, nn.ModuleList):
                for module in child.children():
                    if not mlist:
                        mlist = [module]
                    elif not AutoTP.in_module_list(module, mlist):
                        mlist = mlist + [module]
            else:
                mlist = mlist + AutoTP.get_module_list(child)
        return mlist

    def supported(self):
        unsupported = ['codegen', 'deberta', 'flaubert', 'fsmt', 'gpt2', 'led', 'longformer', 'xlm', 'xlnet']
        self = str(self)
        key = re.search(r": (.*?)Model", self)
        if key is None:
            key = re.search(r": (.*?)Stack", self)
        if key is None:
            key = re.match(r"(.*?)Model", self)
        assert key is not None, "Not able to determine model policy automatically. Please provide policy."
        return key[1].lower() not in unsupported

    def get_layers(self, module):
        layer_list = []
        for key, submodule in module._modules.items():
            if isinstance(submodule, nn.Linear):
                layer_list = layer_list + [f"{self}.{key}"]
            elif isinstance(submodule, nn.LayerNorm) or key == 'LayerNorm' or key == 'layer_norm':
                layer_list = layer_list + ["ln"]
            else:
                layer_list = layer_list + AutoTP.get_layers(key, submodule)
        return layer_list

    def update_policy_list(self, new_module, new_gems):
        if len(self):
            for i, policy in enumerate(self):
                # if module already exists in policy, combine gems and remove duplicates
                if policy[0] == type(new_module):
                    new_gems = set(new_gems + policy[1])
                    self[i] = type(new_module), new_gems
                    return self
        self.append((type(new_module), new_gems))
        return self

    def kernel_supported(self):
        policy = []
        for plcy in replace_policies:
            # instantiate a throw-away policy in order to populate the _orig_layer_class
            _ = plcy(None)
            if isinstance(plcy._orig_layer_class, list):
                policy.extend(iter(plcy._orig_layer_class))
            elif plcy._orig_layer_class is not None:
                policy.append(plcy._orig_layer_class)
        return any(child.__class__ in policy for child in self)

    def tp_parser(self):
        policy_list = []
        module_list = []
        layer_list = []
        gem_list = []

        module_list = AutoTP.get_module_list(self)
        assert AutoTP.supported(self), (
            "AutoTP not supported for model. Please use kernel injection since container policy for model exists."
            if AutoTP.kernel_supported(module_list)
            else "AutoTP not supported for model. Please provide policy."
        )
        for module in module_list:
            for key, submodule in module._modules.items():
                if isinstance(submodule, nn.Linear):
                    layer_list = layer_list + [f".{key}"]
                elif isinstance(submodule, nn.LayerNorm) or key == 'LayerNorm' or key == 'layer_norm':
                    layer_list = layer_list + ["ln"]
                else:
                    layer_list = layer_list + AutoTP.get_layers(key, submodule)
            for i, layer in enumerate(layer_list):
                if layer == 'ln':
                    if layer_list[i - 1] != 'ln':
                        gem_list = gem_list + [layer_list[i - 1]]
                elif 'out_proj' in layer:
                    gem_list = gem_list + [layer]
            layer_list = []
            if gem_list != []:
                gem_list = list(set(gem_list))
                policy_list = AutoTP.update_policy_list(policy_list, module, gem_list)
                gem_list = []
        assert len(policy_list), "AutoTP not supported for model. Please use kernel injection since container policy for model exists." \
            if AutoTP.kernel_supported(module_list) else "Not able to determine model policy automatically. Please provide policy."
        return policy_list
