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
        self.conv2d86 = Conv2d(36, 864, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d87 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x267, x264):
        x268=self.conv2d86(x267)
        x269=self.sigmoid17(x268)
        x270=operator.mul(x269, x264)
        x271=self.conv2d87(x270)
        x272=self.batchnorm2d51(x271)
        return x272

m = M().eval()
x267 = torch.randn(torch.Size([1, 36, 1, 1]))
x264 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x267, x264)
end = time.time()
print(end-start)
