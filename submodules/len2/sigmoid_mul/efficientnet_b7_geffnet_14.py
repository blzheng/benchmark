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

    def forward(self, x212, x208):
        x213=x212.sigmoid()
        x214=operator.mul(x208, x213)
        return x214

m = M().eval()
x212 = torch.randn(torch.Size([1, 480, 1, 1]))
x208 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x212, x208)
end = time.time()
print(end-start)
