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
        self.sigmoid8 = Sigmoid()

    def forward(self, x199, x195):
        x200=self.sigmoid8(x199)
        x201=operator.mul(x200, x195)
        return x201

m = M().eval()
x199 = torch.randn(torch.Size([1, 960, 1, 1]))
x195 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x199, x195)
end = time.time()
print(end-start)
