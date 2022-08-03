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
        self.conv2d36 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu624 = ReLU6(inplace=True)
        self.conv2d37 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d37 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu625 = ReLU6(inplace=True)

    def forward(self, x103):
        x104=self.conv2d36(x103)
        x105=self.batchnorm2d36(x104)
        x106=self.relu624(x105)
        x107=self.conv2d37(x106)
        x108=self.batchnorm2d37(x107)
        x109=self.relu625(x108)
        return x109

m = M().eval()
x103 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x103)
end = time.time()
print(end-start)
