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
        self.sigmoid3 = Sigmoid()

    def forward(self, x140, x136):
        x141=self.sigmoid3(x140)
        x142=operator.mul(x141, x136)
        return x142

m = M().eval()
x140 = torch.randn(torch.Size([1, 640, 1, 1]))
x136 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x140, x136)
end = time.time()
print(end-start)
