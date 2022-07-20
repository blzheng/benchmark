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
        self.conv2d88 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d89 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x261, x249):
        x262=self.conv2d88(x261)
        x263=self.batchnorm2d52(x262)
        x264=operator.add(x263, x249)
        x265=self.conv2d89(x264)
        return x265

m = M().eval()
x261 = torch.randn(torch.Size([1, 816, 14, 14]))
x249 = torch.randn(torch.Size([1, 136, 14, 14]))
start = time.time()
output = m(x261, x249)
end = time.time()
print(end-start)
