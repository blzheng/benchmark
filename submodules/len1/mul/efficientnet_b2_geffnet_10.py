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

    def forward(self, x151, x156):
        x157=operator.mul(x151, x156)
        return x157

m = M().eval()
x151 = torch.randn(torch.Size([1, 528, 14, 14]))
x156 = torch.randn(torch.Size([1, 528, 1, 1]))
start = time.time()
output = m(x151, x156)
end = time.time()
print(end-start)
