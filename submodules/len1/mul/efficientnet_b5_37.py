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

    def forward(self, x583, x578):
        x584=operator.mul(x583, x578)
        return x584

m = M().eval()
x583 = torch.randn(torch.Size([1, 3072, 1, 1]))
x578 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x583, x578)
end = time.time()
print(end-start)
