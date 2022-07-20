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
        self.conv2d87 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x269, x264):
        x270=operator.mul(x269, x264)
        x271=self.conv2d87(x270)
        x272=self.batchnorm2d51(x271)
        return x272

m = M().eval()
x269 = torch.randn(torch.Size([1, 768, 1, 1]))
x264 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x269, x264)
end = time.time()
print(end-start)