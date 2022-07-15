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

    def forward(self, x812, x798):
        x813=operator.add(x812, x798)
        return x813

m = M().eval()
x812 = torch.randn(torch.Size([1, 640, 7, 7]))
x798 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x812, x798)
end = time.time()
print(end-start)
