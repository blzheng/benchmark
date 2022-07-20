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

    def forward(self, x197, x193):
        x198=x197.sigmoid()
        x199=operator.mul(x193, x198)
        return x199

m = M().eval()
x197 = torch.randn(torch.Size([1, 480, 1, 1]))
x193 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x197, x193)
end = time.time()
print(end-start)
