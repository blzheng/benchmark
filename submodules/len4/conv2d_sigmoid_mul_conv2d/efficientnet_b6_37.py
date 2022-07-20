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
        self.conv2d186 = Conv2d(86, 2064, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid37 = Sigmoid()
        self.conv2d187 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x583, x580):
        x584=self.conv2d186(x583)
        x585=self.sigmoid37(x584)
        x586=operator.mul(x585, x580)
        x587=self.conv2d187(x586)
        return x587

m = M().eval()
x583 = torch.randn(torch.Size([1, 86, 1, 1]))
x580 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x583, x580)
end = time.time()
print(end-start)
