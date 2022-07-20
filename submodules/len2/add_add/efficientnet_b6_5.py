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

    def forward(self, x637, x622, x653):
        x638=operator.add(x637, x622)
        x654=operator.add(x653, x638)
        return x654

m = M().eval()
x637 = torch.randn(torch.Size([1, 344, 7, 7]))
x622 = torch.randn(torch.Size([1, 344, 7, 7]))
x653 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x637, x622, x653)
end = time.time()
print(end-start)
