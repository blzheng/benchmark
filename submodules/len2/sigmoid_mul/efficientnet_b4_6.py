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
        self.sigmoid6 = Sigmoid()

    def forward(self, x99, x95):
        x100=self.sigmoid6(x99)
        x101=operator.mul(x100, x95)
        return x101

m = M().eval()
x99 = torch.randn(torch.Size([1, 192, 1, 1]))
x95 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x99, x95)
end = time.time()
print(end-start)
