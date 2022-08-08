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
        self.conv2d88 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d88 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu85 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d89 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d90 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x292):
        x293=self.conv2d88(x292)
        x294=self.batchnorm2d88(x293)
        x295=self.relu85(x294)
        x296=self.conv2d89(x295)
        x297=self.batchnorm2d89(x296)
        x298=self.relu85(x297)
        x299=self.conv2d90(x298)
        return x299

m = M().eval()
x292 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x292)
end = time.time()
print(end-start)
