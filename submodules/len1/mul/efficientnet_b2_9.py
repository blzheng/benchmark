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

    def forward(self, x144, x139):
        x145=operator.mul(x144, x139)
        return x145

m = M().eval()
x144 = torch.randn(torch.Size([1, 528, 1, 1]))
x139 = torch.randn(torch.Size([1, 528, 14, 14]))
start = time.time()
output = m(x144, x139)
end = time.time()
print(end-start)
