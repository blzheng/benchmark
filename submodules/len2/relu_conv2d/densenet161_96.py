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
        self.relu97 = ReLU(inplace=True)
        self.conv2d97 = Conv2d(1776, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x345):
        x346=self.relu97(x345)
        x347=self.conv2d97(x346)
        return x347

m = M().eval()
x345 = torch.randn(torch.Size([1, 1776, 14, 14]))
start = time.time()
output = m(x345)
end = time.time()
print(end-start)
