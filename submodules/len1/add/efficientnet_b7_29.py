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

    def forward(self, x540, x525):
        x541=operator.add(x540, x525)
        return x541

m = M().eval()
x540 = torch.randn(torch.Size([1, 224, 14, 14]))
x525 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x540, x525)
end = time.time()
print(end-start)
