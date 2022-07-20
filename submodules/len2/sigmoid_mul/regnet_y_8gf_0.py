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

    def forward(self, x15, x11):
        x16=self.sigmoid0(x15)
        x17=operator.mul(x16, x11)
        return x17

m = M().eval()
x15 = torch.randn(torch.Size([1, 224, 1, 1]))
x11 = torch.randn(torch.Size([1, 224, 56, 56]))
start = time.time()
output = m(x15, x11)
end = time.time()
print(end-start)
