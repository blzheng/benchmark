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
        self.sigmoid23 = Sigmoid()

    def forward(self, x387, x383):
        x388=self.sigmoid23(x387)
        x389=operator.mul(x388, x383)
        return x389

m = M().eval()
x387 = torch.randn(torch.Size([1, 336, 1, 1]))
x383 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x387, x383)
end = time.time()
print(end-start)
