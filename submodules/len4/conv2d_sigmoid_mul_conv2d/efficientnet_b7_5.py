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
        self.conv2d25 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d26 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x76, x73):
        x77=self.conv2d25(x76)
        x78=self.sigmoid5(x77)
        x79=operator.mul(x78, x73)
        x80=self.conv2d26(x79)
        return x80

m = M().eval()
x76 = torch.randn(torch.Size([1, 12, 1, 1]))
x73 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x76, x73)
end = time.time()
print(end-start)
