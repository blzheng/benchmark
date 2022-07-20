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
        self.layer_scale26 = torch.rand(torch.Size([384, 1, 1]))

    def forward(self, x311):
        x312=torch.permute(x311, [0, 3, 1, 2])
        x313=operator.mul(self.layer_scale26, x312)
        return x313

m = M().eval()
x311 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x311)
end = time.time()
print(end-start)
