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
        self.conv2d64 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU(inplace=True)
        self.conv2d65 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x210):
        x211=self.conv2d64(x210)
        x212=self.batchnorm2d64(x211)
        x213=self.relu61(x212)
        x214=self.conv2d65(x213)
        return x214

m = M().eval()
x210 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x210)
end = time.time()
print(end-start)
