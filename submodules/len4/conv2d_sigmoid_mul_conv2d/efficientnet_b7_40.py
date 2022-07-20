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
        self.conv2d200 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid40 = Sigmoid()
        self.conv2d201 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x628, x625):
        x629=self.conv2d200(x628)
        x630=self.sigmoid40(x629)
        x631=operator.mul(x630, x625)
        x632=self.conv2d201(x631)
        return x632

m = M().eval()
x628 = torch.randn(torch.Size([1, 96, 1, 1]))
x625 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x628, x625)
end = time.time()
print(end-start)
