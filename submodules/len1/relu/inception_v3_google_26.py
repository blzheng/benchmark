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

    def forward(self, x101):
        x102=torch.nn.functional.relu(x101,inplace=True)
        return x102

m = M().eval()
x101 = torch.randn(torch.Size([1, 384, 12, 12]))
start = time.time()
output = m(x101)
end = time.time()
print(end-start)
