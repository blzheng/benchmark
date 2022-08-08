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

    def forward(self, x68, x64):
        x69=operator.add(x68, x64)
        return x69

m = M().eval()
x68 = torch.randn(torch.Size([1, 384, 768]))
x64 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x68, x64)
end = time.time()
print(end-start)
