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
        self.conv2d46 = Conv2d(16, 384, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()

    def forward(self, x141, x138):
        x142=self.conv2d46(x141)
        x143=self.sigmoid9(x142)
        x144=operator.mul(x143, x138)
        return x144

m = M().eval()
x141 = torch.randn(torch.Size([1, 16, 1, 1]))
x138 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x141, x138)
end = time.time()
print(end-start)
