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

    def forward(self, x158, x144):
        x159=operator.add(x158, x144)
        return x159

m = M().eval()
x158 = torch.randn(torch.Size([1, 64, 28, 28]))
x144 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x158, x144)
end = time.time()
print(end-start)
