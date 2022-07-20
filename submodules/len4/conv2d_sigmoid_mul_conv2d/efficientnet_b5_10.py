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
        self.conv2d51 = Conv2d(16, 384, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d52 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x157, x154):
        x158=self.conv2d51(x157)
        x159=self.sigmoid10(x158)
        x160=operator.mul(x159, x154)
        x161=self.conv2d52(x160)
        return x161

m = M().eval()
x157 = torch.randn(torch.Size([1, 16, 1, 1]))
x154 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x157, x154)
end = time.time()
print(end-start)
