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
        self.sigmoid16 = Sigmoid()

    def forward(self, x379, x375):
        x380=self.sigmoid16(x379)
        x381=operator.mul(x380, x375)
        return x381

m = M().eval()
x379 = torch.randn(torch.Size([1, 1344, 1, 1]))
x375 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x379, x375)
end = time.time()
print(end-start)
