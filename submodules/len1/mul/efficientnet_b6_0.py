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

    def forward(self, x11, x6):
        x12=operator.mul(x11, x6)
        return x12

m = M().eval()
x11 = torch.randn(torch.Size([1, 56, 1, 1]))
x6 = torch.randn(torch.Size([1, 56, 112, 112]))
start = time.time()
output = m(x11, x6)
end = time.time()
print(end-start)