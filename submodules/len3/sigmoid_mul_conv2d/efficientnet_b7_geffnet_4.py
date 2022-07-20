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
        self.conv2d21 = Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x64, x60):
        x65=x64.sigmoid()
        x66=operator.mul(x60, x65)
        x67=self.conv2d21(x66)
        return x67

m = M().eval()
x64 = torch.randn(torch.Size([1, 192, 1, 1]))
x60 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x64, x60)
end = time.time()
print(end-start)
