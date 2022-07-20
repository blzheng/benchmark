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

    def forward(self, x6):
        x7=operator.mul(x6, 0.448)
        x8=operator.add(x7, -0.08799999999999997)
        return x8

m = M().eval()
x6 = torch.randn(torch.Size([1, 1, 224, 224]))
start = time.time()
output = m(x6)
end = time.time()
print(end-start)
