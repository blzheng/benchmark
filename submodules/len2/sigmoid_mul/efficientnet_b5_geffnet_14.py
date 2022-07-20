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

    def forward(self, x213, x209):
        x214=x213.sigmoid()
        x215=operator.mul(x209, x214)
        return x215

m = M().eval()
x213 = torch.randn(torch.Size([1, 768, 1, 1]))
x209 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x213, x209)
end = time.time()
print(end-start)