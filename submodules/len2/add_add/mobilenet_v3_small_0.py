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

    def forward(self, x59, x45, x74):
        x60=operator.add(x59, x45)
        x75=operator.add(x74, x60)
        return x75

m = M().eval()
x59 = torch.randn(torch.Size([1, 40, 14, 14]))
x45 = torch.randn(torch.Size([1, 40, 14, 14]))
x74 = torch.randn(torch.Size([1, 40, 14, 14]))
start = time.time()
output = m(x59, x45, x74)
end = time.time()
print(end-start)
