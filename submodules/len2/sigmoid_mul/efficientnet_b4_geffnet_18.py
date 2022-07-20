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

    def forward(self, x274, x270):
        x275=x274.sigmoid()
        x276=operator.mul(x270, x275)
        return x276

m = M().eval()
x274 = torch.randn(torch.Size([1, 960, 1, 1]))
x270 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x274, x270)
end = time.time()
print(end-start)
