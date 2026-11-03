"""Real-pair affine rotation scan: O(BTH) work/storage, O(log T) scan depth."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from src.neural_ssm.utils.scan import associative_scan


def rotate(pair, vector):
    """Multiply (r + i*s) by (x + i*y) using real arithmetic."""
    r, s = pair.unbind(-1)
    x, y = vector.unbind(-1)
    return torch.stack((r * x - s * y, s * x + r * y), -1)


def compose(first, second):
    a1, b1 = first
    a2, b2 = second
    return rotate(a2, a1), rotate(a2, b1) + b2


def bounded_proposal(z, radius):
    # Algebraically M*z/sqrt(1+||z||^2), without squaring huge raw values.
    scale = z.abs().amax(-1, keepdim=True).clamp_min(1)
    scaled = z / scale
    denominator = torch.sqrt(scale.reciprocal().square() + scaled.square().sum(-1, keepdim=True))
    return radius * scaled / denominator


def memory_scan(a, b, initial):
    if a.shape[1] == 0:
        return b
    prefixes, writes = associative_scan(compose, (a, b), axis=1)
    return rotate(prefixes, initial[:, None]) + writes


def memory_reference(a, b, initial):
    current, values = initial, []
    for index in range(a.shape[1]):
        current = rotate(a[:, index], current) + b[:, index]
        values.append(current)
    return torch.stack(values, 1) if values else b


class BoundedMemory(nn.Module):
    def __init__(self, d_input, config):
        super().__init__()
        self.heads, self.radius = config.heads, config.memory_radius
        self.gate = nn.Linear(d_input, config.heads)
        self.angle = nn.Linear(d_input, config.heads)
        self.write = nn.Linear(d_input, 2 * config.heads)
        with torch.no_grad():
            self.gate.weight.normal_(0, .05 / math.sqrt(d_input))
            self.gate.bias.fill_(math.log(config.initial_retention / (1 - config.initial_retention)))
            self.angle.weight.normal_(0, .05 / math.sqrt(d_input))
            self.angle.bias.zero_()
            self.write.weight.normal_(0, .3 / math.sqrt(d_input))
            self.write.bias.zero_()

    def coefficients(self, x, valid):
        # Caller disables autocast and promotes inputs/weights to >=FP32.
        linear = lambda layer: F.linear(x, layer.weight.to(x.dtype), layer.bias.to(x.dtype))
        retention = torch.sigmoid(linear(self.gate))
        angle = math.pi * torch.tanh(linear(self.angle))
        proposal = bounded_proposal(linear(self.write).reshape(*x.shape[:2], self.heads, 2), self.radius)
        a = retention[..., None] * torch.stack((angle.cos(), angle.sin()), -1)
        b = (1 - retention[..., None]) * proposal
        identity = torch.stack((torch.ones_like(retention), torch.zeros_like(retention)), -1)
        return torch.where(valid[..., None, None], a, identity), torch.where(valid[..., None, None], b, 0)
