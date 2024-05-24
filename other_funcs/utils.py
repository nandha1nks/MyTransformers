from transformers import Conv1D
import torch.nn as nn


def convert_transformers_conv1d_to_linear(module):
    for name, child in module.named_children():
        if isinstance(child, Conv1D):
            lin = nn.Linear(child.weight.shape[0],
                            child.weight.shape[1],
                            bias=child.bias is not None)
            lin.weight = nn.Parameter(child.weight.T.contiguous())
            if child.bias is not None:
                lin.bias = child.bias
            setattr(module, name, lin)
        else:
            convert_transformers_conv1d_to_linear(child)
