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
        self.relu52 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x201, x215):
        x216=operator.add(x201, x215)
        x217=self.relu52(x216)
        x218=self.conv2d69(x217)
        return x218

m = M().eval()
x201 = torch.randn(torch.Size([1, 336, 14, 14]))
x215 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x201, x215)
end = time.time()
print(end-start)
