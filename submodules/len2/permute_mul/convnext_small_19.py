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
        self.layer_scale19 = torch.rand(torch.Size([384, 1, 1])).to(torch.float32)

    def forward(self, x234):
        x235=torch.permute(x234, [0, 3, 1, 2])
        x236=operator.mul(self.layer_scale19, x235)
        return x236

m = M().eval()
x234 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x234)
end = time.time()
print(end-start)
