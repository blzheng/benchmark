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
        self.sigmoid27 = Sigmoid()

    def forward(self, x425, x421):
        x426=self.sigmoid27(x425)
        x427=operator.mul(x426, x421)
        return x427

m = M().eval()
x425 = torch.randn(torch.Size([1, 960, 1, 1]))
x421 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x425, x421)
end = time.time()
print(end-start)
