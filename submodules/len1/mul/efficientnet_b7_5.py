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

    def forward(self, x78, x73):
        x79=operator.mul(x78, x73)
        return x79

m = M().eval()
x78 = torch.randn(torch.Size([1, 288, 1, 1]))
x73 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x78, x73)
end = time.time()
print(end-start)
