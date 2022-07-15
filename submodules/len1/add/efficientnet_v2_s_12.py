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

    def forward(self, x158, x143):
        x159=operator.add(x158, x143)
        return x159

m = M().eval()
x158 = torch.randn(torch.Size([1, 128, 14, 14]))
x143 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x158, x143)
end = time.time()
print(end-start)
