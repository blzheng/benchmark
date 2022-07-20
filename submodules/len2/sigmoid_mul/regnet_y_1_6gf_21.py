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
        self.sigmoid21 = Sigmoid()

    def forward(self, x355, x351):
        x356=self.sigmoid21(x355)
        x357=operator.mul(x356, x351)
        return x357

m = M().eval()
x355 = torch.randn(torch.Size([1, 336, 1, 1]))
x351 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x355, x351)
end = time.time()
print(end-start)
