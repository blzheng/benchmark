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

    def forward(self, x185, x181):
        x186=x185.sigmoid()
        x187=operator.mul(x181, x186)
        return x187

m = M().eval()
x185 = torch.randn(torch.Size([1, 480, 1, 1]))
x181 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x185, x181)
end = time.time()
print(end-start)
