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
        self.conv2d30 = Conv2d(120, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d31 = Conv2d(48, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x87):
        x88=self.conv2d30(x87)
        x89=self.batchnorm2d20(x88)
        x90=self.conv2d31(x89)
        x91=self.batchnorm2d21(x90)
        return x91

m = M().eval()
x87 = torch.randn(torch.Size([1, 120, 14, 14]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
