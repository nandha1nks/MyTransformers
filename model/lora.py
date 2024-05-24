import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLayer(nn.Module):
    def __init__(self, alpha, rank, old_layer: nn.Linear):
        super().__init__()
        self.scale = alpha/rank
        self.A = nn.Linear(old_layer.in_features, rank,  bias=False)
        self.B = nn.Linear(rank, old_layer.out_features)
        self.B.weight = nn.Parameter(torch.zeros_like(self.B.weight))

        self.old_layer = old_layer

    def forward(self, x):
        y = self.old_layer(x)
        return y + self.scale * self.B(self.A(x))


def get_trainable_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def replace_linear_layers_with_lora(model, alpha, rank):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            lora = LoraLayer(alpha, rank, child)
            setattr(model, name, lora)
        else:
            replace_linear_layers_with_lora(child, alpha, rank)


def make_model_for_lora(model: nn.Module, alpha, rank, layers_to_change=None):
    get_trainable_parameters(model)
    for p in model.parameters():
        p.requires_grad = False
    get_trainable_parameters(model)
    replace_linear_layers_with_lora(model, alpha, rank)
    get_trainable_parameters(model)
