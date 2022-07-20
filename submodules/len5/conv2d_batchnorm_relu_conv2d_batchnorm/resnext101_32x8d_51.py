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
        self.conv2d80 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d80 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu76 = ReLU(inplace=True)
        self.conv2d81 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d81 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x263):
        x264=self.conv2d80(x263)
        x265=self.batchnorm2d80(x264)
        x266=self.relu76(x265)
        x267=self.conv2d81(x266)
        x268=self.batchnorm2d81(x267)
        return x268

m = M().eval()
x263 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x263)
end = time.time()
print(end-start)
