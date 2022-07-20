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
        self.conv2d68 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x215, x211):
        x216=self.sigmoid9(x215)
        x217=operator.mul(x216, x211)
        x218=self.conv2d68(x217)
        return x218

m = M().eval()
x215 = torch.randn(torch.Size([1, 960, 1, 1]))
x211 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x215, x211)
end = time.time()
print(end-start)
