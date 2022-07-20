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
        self.sigmoid9 = Sigmoid()

    def forward(self, x142, x138):
        x143=self.sigmoid9(x142)
        x144=operator.mul(x143, x138)
        return x144

m = M().eval()
x142 = torch.randn(torch.Size([1, 384, 1, 1]))
x138 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x142, x138)
end = time.time()
print(end-start)
