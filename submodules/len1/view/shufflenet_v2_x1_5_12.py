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

    def forward(self, x152, x154, x158, x156, x157):
        x159=x152.view(x154, 2, x158, x156, x157)
        return x159

m = M().eval()
x152 = torch.randn(torch.Size([1, 352, 14, 14]))
x154 = 1
x158 = 176
x156 = 14
x157 = 14
start = time.time()
output = m(x152, x154, x158, x156, x157)
end = time.time()
print(end-start)
