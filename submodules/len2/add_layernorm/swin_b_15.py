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
        self.layernorm21 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)

    def forward(self, x218, x225):
        x226=operator.add(x218, x225)
        x227=self.layernorm21(x226)
        return x227

m = M().eval()
x218 = torch.randn(torch.Size([1, 14, 14, 512]))
x225 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x218, x225)
end = time.time()
print(end-start)
