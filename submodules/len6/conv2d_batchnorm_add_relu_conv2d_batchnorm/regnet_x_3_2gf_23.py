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
        self.conv2d66 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu63 = ReLU(inplace=True)
        self.conv2d67 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x215, x209):
        x216=self.conv2d66(x215)
        x217=self.batchnorm2d66(x216)
        x218=operator.add(x209, x217)
        x219=self.relu63(x218)
        x220=self.conv2d67(x219)
        x221=self.batchnorm2d67(x220)
        return x221

m = M().eval()
x215 = torch.randn(torch.Size([1, 432, 14, 14]))
x209 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x215, x209)
end = time.time()
print(end-start)
