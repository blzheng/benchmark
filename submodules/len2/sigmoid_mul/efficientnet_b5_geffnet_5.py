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

    def forward(self, x80, x76):
        x81=x80.sigmoid()
        x82=operator.mul(x76, x81)
        return x82

m = M().eval()
x80 = torch.randn(torch.Size([1, 240, 1, 1]))
x76 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x80, x76)
end = time.time()
print(end-start)
