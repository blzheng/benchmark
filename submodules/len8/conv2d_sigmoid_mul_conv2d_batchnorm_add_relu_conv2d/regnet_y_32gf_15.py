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
        self.conv2d82 = Conv2d(348, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d83 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)
        self.conv2d84 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x258, x255, x249):
        x259=self.conv2d82(x258)
        x260=self.sigmoid15(x259)
        x261=operator.mul(x260, x255)
        x262=self.conv2d83(x261)
        x263=self.batchnorm2d51(x262)
        x264=operator.add(x249, x263)
        x265=self.relu64(x264)
        x266=self.conv2d84(x265)
        return x266

m = M().eval()
x258 = torch.randn(torch.Size([1, 348, 1, 1]))
x255 = torch.randn(torch.Size([1, 1392, 14, 14]))
x249 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x258, x255, x249)
end = time.time()
print(end-start)
