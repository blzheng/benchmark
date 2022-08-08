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
        self.relu10 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d17 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x49):
        x50=self.relu10(x49)
        x51=self.conv2d15(x50)
        x52=self.batchnorm2d15(x51)
        x53=self.relu13(x52)
        x54=self.conv2d16(x53)
        x55=self.batchnorm2d16(x54)
        x56=self.relu13(x55)
        x57=self.conv2d17(x56)
        x58=self.batchnorm2d17(x57)
        return x58

m = M().eval()
x49 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x49)
end = time.time()
print(end-start)
