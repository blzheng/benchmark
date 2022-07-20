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
        self.batchnorm2d51 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)
        self.conv2d84 = Conv2d(896, 2016, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d52 = BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x262, x249):
        x263=self.batchnorm2d51(x262)
        x264=operator.add(x249, x263)
        x265=self.relu64(x264)
        x266=self.conv2d84(x265)
        x267=self.batchnorm2d52(x266)
        return x267

m = M().eval()
x262 = torch.randn(torch.Size([1, 896, 14, 14]))
x249 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x262, x249)
end = time.time()
print(end-start)
