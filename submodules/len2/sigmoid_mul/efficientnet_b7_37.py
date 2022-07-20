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
        self.sigmoid37 = Sigmoid()

    def forward(self, x583, x579):
        x584=self.sigmoid37(x583)
        x585=operator.mul(x584, x579)
        return x585

m = M().eval()
x583 = torch.randn(torch.Size([1, 1344, 1, 1]))
x579 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x583, x579)
end = time.time()
print(end-start)
