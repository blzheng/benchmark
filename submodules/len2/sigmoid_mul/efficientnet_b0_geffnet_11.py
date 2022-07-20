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

    def forward(self, x171, x167):
        x172=x171.sigmoid()
        x173=operator.mul(x167, x172)
        return x173

m = M().eval()
x171 = torch.randn(torch.Size([1, 672, 1, 1]))
x167 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x171, x167)
end = time.time()
print(end-start)
