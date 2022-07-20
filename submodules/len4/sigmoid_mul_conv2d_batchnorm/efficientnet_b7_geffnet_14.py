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
        self.conv2d71 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x212, x208):
        x213=x212.sigmoid()
        x214=operator.mul(x208, x213)
        x215=self.conv2d71(x214)
        x216=self.batchnorm2d41(x215)
        return x216

m = M().eval()
x212 = torch.randn(torch.Size([1, 480, 1, 1]))
x208 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x212, x208)
end = time.time()
print(end-start)
