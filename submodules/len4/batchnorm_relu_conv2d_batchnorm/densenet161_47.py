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
        self.batchnorm2d97 = BatchNorm2d(1776, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu97 = ReLU(inplace=True)
        self.conv2d97 = Conv2d(1776, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d98 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x344):
        x345=self.batchnorm2d97(x344)
        x346=self.relu97(x345)
        x347=self.conv2d97(x346)
        x348=self.batchnorm2d98(x347)
        return x348

m = M().eval()
x344 = torch.randn(torch.Size([1, 1776, 14, 14]))
start = time.time()
output = m(x344)
end = time.time()
print(end-start)
