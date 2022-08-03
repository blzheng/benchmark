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
        self.relu632 = ReLU6(inplace=True)
        self.conv2d49 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d49 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu633 = ReLU6(inplace=True)
        self.conv2d50 = Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x140):
        x141=self.relu632(x140)
        x142=self.conv2d49(x141)
        x143=self.batchnorm2d49(x142)
        x144=self.relu633(x143)
        x145=self.conv2d50(x144)
        x146=self.batchnorm2d50(x145)
        x147=self.conv2d51(x146)
        x148=self.batchnorm2d51(x147)
        return x148

m = M().eval()
x140 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x140)
end = time.time()
print(end-start)
