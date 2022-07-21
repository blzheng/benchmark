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
        self.relu49 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d57 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x179):
        x180=self.relu49(x179)
        x181=self.conv2d55(x180)
        x182=self.batchnorm2d55(x181)
        x183=self.relu52(x182)
        x184=self.conv2d56(x183)
        x185=self.batchnorm2d56(x184)
        x186=self.relu52(x185)
        x187=self.conv2d57(x186)
        x188=self.batchnorm2d57(x187)
        return x188

m = M().eval()
x179 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x179)
end = time.time()
print(end-start)
