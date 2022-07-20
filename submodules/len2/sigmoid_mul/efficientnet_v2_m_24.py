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
        self.sigmoid24 = Sigmoid()

    def forward(self, x472, x468):
        x473=self.sigmoid24(x472)
        x474=operator.mul(x473, x468)
        return x474

m = M().eval()
x472 = torch.randn(torch.Size([1, 1824, 1, 1]))
x468 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x472, x468)
end = time.time()
print(end-start)
