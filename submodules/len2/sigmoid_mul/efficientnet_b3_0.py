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
        self.sigmoid0 = Sigmoid()

    def forward(self, x10, x6):
        x11=self.sigmoid0(x10)
        x12=operator.mul(x11, x6)
        return x12

m = M().eval()
x10 = torch.randn(torch.Size([1, 40, 1, 1]))
x6 = torch.randn(torch.Size([1, 40, 112, 112]))
start = time.time()
output = m(x10, x6)
end = time.time()
print(end-start)
