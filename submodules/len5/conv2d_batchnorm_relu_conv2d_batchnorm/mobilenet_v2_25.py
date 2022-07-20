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
        self.conv2d37 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d37 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu625 = ReLU6(inplace=True)
        self.conv2d38 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x106):
        x107=self.conv2d37(x106)
        x108=self.batchnorm2d37(x107)
        x109=self.relu625(x108)
        x110=self.conv2d38(x109)
        x111=self.batchnorm2d38(x110)
        return x111

m = M().eval()
x106 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x106)
end = time.time()
print(end-start)
