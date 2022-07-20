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
        self.relu13 = ReLU()
        self.conv2d49 = Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid8 = Hardsigmoid()

    def forward(self, x141, x139):
        x142=self.relu13(x141)
        x143=self.conv2d49(x142)
        x144=self.hardsigmoid8(x143)
        x145=operator.mul(x144, x139)
        return x145

m = M().eval()
x141 = torch.randn(torch.Size([1, 144, 1, 1]))
x139 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x141, x139)
end = time.time()
print(end-start)
