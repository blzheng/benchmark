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
        self.sigmoid0 = Sigmoid()

    def forward(self, x94, x90):
        x95=self.sigmoid0(x94)
        x96=operator.mul(x95, x90)
        return x96

m = M().eval()
x94 = torch.randn(torch.Size([1, 320, 1, 1]))
x90 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x94, x90)
end = time.time()
print(end-start)
