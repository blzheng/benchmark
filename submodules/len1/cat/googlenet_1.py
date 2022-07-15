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

    def forward(self, x27, x33, x39, x43):
        x44=torch.cat([x27, x33, x39, x43], 1)
        return x44

m = M().eval()
x27 = torch.randn(torch.Size([1, 64, 28, 28]))
x33 = torch.randn(torch.Size([1, 128, 28, 28]))
x39 = torch.randn(torch.Size([1, 32, 28, 28]))
x43 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x27, x33, x39, x43)
end = time.time()
print(end-start)
