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
        self.sigmoid10 = Sigmoid()

    def forward(self, x157, x153):
        x158=self.sigmoid10(x157)
        x159=operator.mul(x158, x153)
        return x159

m = M().eval()
x157 = torch.randn(torch.Size([1, 288, 1, 1]))
x153 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x157, x153)
end = time.time()
print(end-start)
