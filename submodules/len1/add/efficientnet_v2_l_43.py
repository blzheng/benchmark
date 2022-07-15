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

    def forward(self, x606, x591):
        x607=operator.add(x606, x591)
        return x607

m = M().eval()
x606 = torch.randn(torch.Size([1, 384, 7, 7]))
x591 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x606, x591)
end = time.time()
print(end-start)
