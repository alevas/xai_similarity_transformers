# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import logging
import torch
import torch.nn as nn
import configs


class LayerNormXAI(nn.Module):
    __constants__ = ['weight', 'bias', 'eps']

    def __init__(self, hidden, eps=1e-5, elementwise_affine=True, args=None, dtype=None):
        factory_kwargs = {'device': configs.device, 'dtype': dtype}
        super(LayerNormXAI, self).__init__()
        self.sigma = args.sigma
        self.hidden = hidden
        self.adanorm_scale = args.adanorm_scale
        self.nowb_scale = args.nowb_scale
        self.mean_detach = args.mean_detach
        self.std_detach = args.std_detach
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.hidden, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.hidden, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):

        mean = input.mean(dim=-1, keepdim=True)
        std = torch.std(input, dim=-1, keepdim=True,
                        unbiased=False)  # xai_impl unbiased deactivates Bessel's correction
        std_real = torch.sqrt(((input - mean) ** 2).sum(dim=-1, keepdims=True) / input.shape[-1])
        if not torch.all(std.eq(std_real)):
            logging.debug("STD calculation if off!")
        std = torch.sqrt(((input - mean) ** 2).sum(dim=-1, keepdims=True) / input.shape[-1])

        if self.std_detach:
            std = std.detach()  # xai_impl
        input_norm = (input - mean) / (std + self.eps)

        input_norm = input_norm * self.weight + self.bias

        return input_norm

