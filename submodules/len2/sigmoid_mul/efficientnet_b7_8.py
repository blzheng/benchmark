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
        self.sigmoid8 = Sigmoid()

    def forward(self, x125, x121):
        x126=self.sigmoid8(x125)
        x127=operator.mul(x126, x121)
        return x127

m = M().eval()
x125 = torch.randn(torch.Size([1, 288, 1, 1]))
x121 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x125, x121)
end = time.time()
print(end-start)
