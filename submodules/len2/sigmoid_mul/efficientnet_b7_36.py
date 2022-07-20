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
        self.sigmoid36 = Sigmoid()

    def forward(self, x567, x563):
        x568=self.sigmoid36(x567)
        x569=operator.mul(x568, x563)
        return x569

m = M().eval()
x567 = torch.randn(torch.Size([1, 1344, 1, 1]))
x563 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x567, x563)
end = time.time()
print(end-start)
