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
        self.batchnorm2d4 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x15, x11):
        x16=self.batchnorm2d4(x15)
        x17=operator.add(x16, x11)
        x18=self.relu3(x17)
        x19=self.conv2d5(x18)
        x20=self.batchnorm2d5(x19)
        return x20

m = M().eval()
x15 = torch.randn(torch.Size([1, 64, 56, 56]))
x11 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x15, x11)
end = time.time()
print(end-start)
