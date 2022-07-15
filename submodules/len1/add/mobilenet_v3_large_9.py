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

    def forward(self, x177, x163):
        x178=operator.add(x177, x163)
        return x178

m = M().eval()
x177 = torch.randn(torch.Size([1, 160, 7, 7]))
x163 = torch.randn(torch.Size([1, 160, 7, 7]))
start = time.time()
output = m(x177, x163)
end = time.time()
print(end-start)
