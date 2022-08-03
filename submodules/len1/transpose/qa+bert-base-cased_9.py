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

    def forward(self, x413):
        x425=x413.transpose(-1, -2)
        return x425

m = M().eval()
x413 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x413)
end = time.time()
print(end-start)
