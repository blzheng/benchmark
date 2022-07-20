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
        self.conv2d84 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)

    def forward(self, x262, x257, x251):
        x263=operator.mul(x262, x257)
        x264=self.conv2d84(x263)
        x265=self.batchnorm2d52(x264)
        x266=operator.add(x251, x265)
        x267=self.relu64(x266)
        return x267

m = M().eval()
x262 = torch.randn(torch.Size([1, 440, 1, 1]))
x257 = torch.randn(torch.Size([1, 440, 7, 7]))
x251 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x262, x257, x251)
end = time.time()
print(end-start)
