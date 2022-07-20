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

    def forward(self, x155, x151):
        x156=x155.sigmoid()
        x157=operator.mul(x151, x156)
        return x157

m = M().eval()
x155 = torch.randn(torch.Size([1, 480, 1, 1]))
x151 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x155, x151)
end = time.time()
print(end-start)
