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
        self.conv2d16 = Conv2d(132, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d17 = Conv2d(1056, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x48, x45):
        x49=self.conv2d16(x48)
        x50=self.sigmoid2(x49)
        x51=operator.mul(x50, x45)
        x52=self.conv2d17(x51)
        return x52

m = M().eval()
x48 = torch.randn(torch.Size([1, 132, 1, 1]))
x45 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x48, x45)
end = time.time()
print(end-start)
