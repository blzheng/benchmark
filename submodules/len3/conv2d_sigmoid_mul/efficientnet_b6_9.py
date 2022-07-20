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
        self.conv2d46 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()

    def forward(self, x143, x140):
        x144=self.conv2d46(x143)
        x145=self.sigmoid9(x144)
        x146=operator.mul(x145, x140)
        return x146

m = M().eval()
x143 = torch.randn(torch.Size([1, 10, 1, 1]))
x140 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x143, x140)
end = time.time()
print(end-start)
