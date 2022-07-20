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
        self.conv2d89 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d89 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu85 = ReLU(inplace=True)
        self.conv2d90 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d90 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x293):
        x294=self.conv2d89(x293)
        x295=self.batchnorm2d89(x294)
        x296=self.relu85(x295)
        x297=self.conv2d90(x296)
        x298=self.batchnorm2d90(x297)
        return x298

m = M().eval()
x293 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x293)
end = time.time()
print(end-start)
