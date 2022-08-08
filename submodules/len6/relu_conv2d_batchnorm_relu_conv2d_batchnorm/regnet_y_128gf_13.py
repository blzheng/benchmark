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
        self.relu64 = ReLU(inplace=True)
        self.conv2d84 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)
        self.conv2d85 = Conv2d(2904, 2904, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)
        self.batchnorm2d53 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x264):
        x265=self.relu64(x264)
        x266=self.conv2d84(x265)
        x267=self.batchnorm2d52(x266)
        x268=self.relu65(x267)
        x269=self.conv2d85(x268)
        x270=self.batchnorm2d53(x269)
        return x270

m = M().eval()
x264 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x264)
end = time.time()
print(end-start)
