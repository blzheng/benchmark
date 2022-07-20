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

    def forward(self, x568, x564):
        x569=self.sigmoid36(x568)
        x570=operator.mul(x569, x564)
        return x570

m = M().eval()
x568 = torch.randn(torch.Size([1, 1824, 1, 1]))
x564 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x568, x564)
end = time.time()
print(end-start)
