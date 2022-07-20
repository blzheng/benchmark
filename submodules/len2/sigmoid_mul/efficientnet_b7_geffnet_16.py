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

    def forward(self, x242, x238):
        x243=x242.sigmoid()
        x244=operator.mul(x238, x243)
        return x244

m = M().eval()
x242 = torch.randn(torch.Size([1, 480, 1, 1]))
x238 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x242, x238)
end = time.time()
print(end-start)
