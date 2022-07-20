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

    def forward(self, x67, x63):
        x68=x67.sigmoid()
        x69=operator.mul(x63, x68)
        return x69

m = M().eval()
x67 = torch.randn(torch.Size([1, 144, 1, 1]))
x63 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x67, x63)
end = time.time()
print(end-start)