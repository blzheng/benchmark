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

    def forward(self, x189, x195, x201, x205):
        x206=torch.cat([x189, x195, x201, x205], 1)
        return x206

m = M().eval()
x189 = torch.randn(torch.Size([1, 384, 7, 7]))
x195 = torch.randn(torch.Size([1, 384, 7, 7]))
x201 = torch.randn(torch.Size([1, 128, 7, 7]))
x205 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x189, x195, x201, x205)
end = time.time()
print(end-start)
