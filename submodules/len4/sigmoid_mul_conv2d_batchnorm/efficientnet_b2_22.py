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
        self.sigmoid22 = Sigmoid()
        self.conv2d113 = Conv2d(2112, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x345, x341):
        x346=self.sigmoid22(x345)
        x347=operator.mul(x346, x341)
        x348=self.conv2d113(x347)
        x349=self.batchnorm2d67(x348)
        return x349

m = M().eval()
x345 = torch.randn(torch.Size([1, 2112, 1, 1]))
x341 = torch.randn(torch.Size([1, 2112, 7, 7]))
start = time.time()
output = m(x345, x341)
end = time.time()
print(end-start)
