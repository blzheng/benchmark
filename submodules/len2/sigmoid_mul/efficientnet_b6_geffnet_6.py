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

    def forward(self, x95, x91):
        x96=x95.sigmoid()
        x97=operator.mul(x91, x96)
        return x97

m = M().eval()
x95 = torch.randn(torch.Size([1, 240, 1, 1]))
x91 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x95, x91)
end = time.time()
print(end-start)
