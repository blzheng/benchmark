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

    def forward(self, x211, x196, x227):
        x212=operator.add(x211, x196)
        x228=operator.add(x227, x212)
        return x228

m = M().eval()
x211 = torch.randn(torch.Size([1, 72, 28, 28]))
x196 = torch.randn(torch.Size([1, 72, 28, 28]))
x227 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x211, x196, x227)
end = time.time()
print(end-start)
