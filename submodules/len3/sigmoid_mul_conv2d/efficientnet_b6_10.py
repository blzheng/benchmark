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
        self.sigmoid10 = Sigmoid()
        self.conv2d52 = Conv2d(432, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x158, x154):
        x159=self.sigmoid10(x158)
        x160=operator.mul(x159, x154)
        x161=self.conv2d52(x160)
        return x161

m = M().eval()
x158 = torch.randn(torch.Size([1, 432, 1, 1]))
x154 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x158, x154)
end = time.time()
print(end-start)
