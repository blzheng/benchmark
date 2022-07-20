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

    def forward(self, x348, x344):
        x349=x348.sigmoid()
        x350=operator.mul(x344, x349)
        return x350

m = M().eval()
x348 = torch.randn(torch.Size([1, 1392, 1, 1]))
x344 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x348, x344)
end = time.time()
print(end-start)