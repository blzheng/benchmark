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

    def forward(self, x200, x196):
        x201=x200.sigmoid()
        x202=operator.mul(x196, x201)
        return x202

m = M().eval()
x200 = torch.randn(torch.Size([1, 1152, 1, 1]))
x196 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x200, x196)
end = time.time()
print(end-start)
