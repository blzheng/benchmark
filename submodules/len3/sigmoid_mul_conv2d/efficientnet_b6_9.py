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
        self.sigmoid9 = Sigmoid()
        self.conv2d47 = Conv2d(240, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x144, x140):
        x145=self.sigmoid9(x144)
        x146=operator.mul(x145, x140)
        x147=self.conv2d47(x146)
        return x147

m = M().eval()
x144 = torch.randn(torch.Size([1, 240, 1, 1]))
x140 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x144, x140)
end = time.time()
print(end-start)
