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
        self.conv2d113 = Conv2d(960, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d114 = Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x349, x345):
        x350=self.sigmoid22(x349)
        x351=operator.mul(x350, x345)
        x352=self.conv2d113(x351)
        x353=self.batchnorm2d67(x352)
        x354=self.conv2d114(x353)
        x355=self.batchnorm2d68(x354)
        return x355

m = M().eval()
x349 = torch.randn(torch.Size([1, 960, 1, 1]))
x345 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x349, x345)
end = time.time()
print(end-start)
