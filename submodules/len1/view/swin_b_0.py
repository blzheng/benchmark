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

    def forward(self, x7):
        x8=x7.view(49, 49, -1)
        return x8

m = M().eval()
x7 = torch.randn(torch.Size([2401, 4]))
start = time.time()
output = m(x7)
end = time.time()
print(end-start)
