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
        self.sigmoid15 = Sigmoid()

    def forward(self, x237, x233):
        x238=self.sigmoid15(x237)
        x239=operator.mul(x238, x233)
        return x239

m = M().eval()
x237 = torch.randn(torch.Size([1, 672, 1, 1]))
x233 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x237, x233)
end = time.time()
print(end-start)
