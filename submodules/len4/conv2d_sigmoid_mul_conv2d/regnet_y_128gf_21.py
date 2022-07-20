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
        self.conv2d112 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()
        self.conv2d113 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x354, x351):
        x355=self.conv2d112(x354)
        x356=self.sigmoid21(x355)
        x357=operator.mul(x356, x351)
        x358=self.conv2d113(x357)
        return x358

m = M().eval()
x354 = torch.randn(torch.Size([1, 726, 1, 1]))
x351 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x354, x351)
end = time.time()
print(end-start)
