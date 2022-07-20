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
        self.conv2d97 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d98 = Conv2d(960, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x310, x307):
        x311=self.conv2d97(x310)
        x312=self.sigmoid15(x311)
        x313=operator.mul(x312, x307)
        x314=self.conv2d98(x313)
        x315=self.batchnorm2d66(x314)
        return x315

m = M().eval()
x310 = torch.randn(torch.Size([1, 40, 1, 1]))
x307 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x310, x307)
end = time.time()
print(end-start)
