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
        self.relu35 = ReLU()
        self.conv2d46 = Conv2d(264, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d47 = Conv2d(1056, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x143, x141):
        x144=self.relu35(x143)
        x145=self.conv2d46(x144)
        x146=self.sigmoid8(x145)
        x147=operator.mul(x146, x141)
        x148=self.conv2d47(x147)
        return x148

m = M().eval()
x143 = torch.randn(torch.Size([1, 264, 1, 1]))
x141 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x143, x141)
end = time.time()
print(end-start)
