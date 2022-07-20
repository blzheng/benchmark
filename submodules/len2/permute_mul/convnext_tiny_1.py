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
        self.layer_scale1 = torch.rand(torch.Size([96, 1, 1]))

    def forward(self, x24):
        x25=torch.permute(x24, [0, 3, 1, 2])
        x26=operator.mul(self.layer_scale1, x25)
        return x26

m = M().eval()
x24 = torch.randn(torch.Size([1, 56, 56, 96]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)
