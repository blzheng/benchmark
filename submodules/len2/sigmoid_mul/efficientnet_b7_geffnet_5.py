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

    def forward(self, x78, x74):
        x79=x78.sigmoid()
        x80=operator.mul(x74, x79)
        return x80

m = M().eval()
x78 = torch.randn(torch.Size([1, 288, 1, 1]))
x74 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x78, x74)
end = time.time()
print(end-start)
