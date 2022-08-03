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

    def forward(self, x62, x64, x68, x66, x67):
        x69=x62.view(x64, 2, x68, x66, x67)
        return x69

m = M().eval()
x62 = torch.randn(torch.Size([1, 244, 28, 28]))
x64 = 1
x68 = 122
x66 = 28
x67 = 28
start = time.time()
output = m(x62, x64, x68, x66, x67)
end = time.time()
print(end-start)
