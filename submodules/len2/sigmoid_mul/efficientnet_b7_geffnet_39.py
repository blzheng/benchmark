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

    def forward(self, x584, x580):
        x585=x584.sigmoid()
        x586=operator.mul(x580, x585)
        return x586

m = M().eval()
x584 = torch.randn(torch.Size([1, 2304, 1, 1]))
x580 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x584, x580)
end = time.time()
print(end-start)
