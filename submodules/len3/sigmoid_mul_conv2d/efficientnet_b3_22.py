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
        self.sigmoid22 = Sigmoid()
        self.conv2d113 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x347, x343):
        x348=self.sigmoid22(x347)
        x349=operator.mul(x348, x343)
        x350=self.conv2d113(x349)
        return x350

m = M().eval()
x347 = torch.randn(torch.Size([1, 1392, 1, 1]))
x343 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x347, x343)
end = time.time()
print(end-start)
