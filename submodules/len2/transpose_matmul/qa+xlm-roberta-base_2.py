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

    def forward(self, x119, x130):
        x131=x119.transpose(-1, -2)
        x132=torch.matmul(x130, x131)
        return x132

m = M().eval()
x119 = torch.randn(torch.Size([1, 12, 384, 64]))
x130 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x119, x130)
end = time.time()
print(end-start)
