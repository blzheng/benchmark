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
        self.conv2d46 = Conv2d(264, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()

    def forward(self, x144, x141):
        x145=self.conv2d46(x144)
        x146=self.sigmoid8(x145)
        x147=operator.mul(x146, x141)
        return x147

m = M().eval()
x144 = torch.randn(torch.Size([1, 264, 1, 1]))
x141 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x144, x141)
end = time.time()
print(end-start)
