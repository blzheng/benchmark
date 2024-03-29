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
        self.sigmoid7 = Sigmoid()

    def forward(self, x109, x105):
        x110=self.sigmoid7(x109)
        x111=operator.mul(x110, x105)
        return x111

m = M().eval()
x109 = torch.randn(torch.Size([1, 288, 1, 1]))
x105 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x109, x105)
end = time.time()
print(end-start)
