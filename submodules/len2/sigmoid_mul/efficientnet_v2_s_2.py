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
        self.sigmoid2 = Sigmoid()

    def forward(self, x105, x101):
        x106=self.sigmoid2(x105)
        x107=operator.mul(x106, x101)
        return x107

m = M().eval()
x105 = torch.randn(torch.Size([1, 512, 1, 1]))
x101 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x105, x101)
end = time.time()
print(end-start)
