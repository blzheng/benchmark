import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x434, x431):
        x435=operator.add(x434, x431)
        x436=self.layernorm2(x435)
        return x436

m = M().eval()
x434 = torch.randn(torch.Size([1, 384, 768]))
x431 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x434, x431)
end = time.time()
print(end-start)
