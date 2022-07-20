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
        self.conv2d19 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x64):
        x65=self.conv2d19(x64)
        x66=self.batchnorm2d19(x65)
        x67=self.relu17(x66)
        x68=self.conv2d20(x67)
        x69=self.batchnorm2d20(x68)
        return x69

m = M().eval()
x64 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x64)
end = time.time()
print(end-start)
