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
        self.sigmoid15 = Sigmoid()

    def forward(self, x238, x234):
        x239=self.sigmoid15(x238)
        x240=operator.mul(x239, x234)
        return x240

m = M().eval()
x238 = torch.randn(torch.Size([1, 432, 1, 1]))
x234 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x238, x234)
end = time.time()
print(end-start)
