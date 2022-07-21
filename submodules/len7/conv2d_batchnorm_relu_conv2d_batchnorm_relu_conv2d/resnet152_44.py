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
        self.conv2d136 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d136 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu133 = ReLU(inplace=True)
        self.conv2d137 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d137 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d138 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x450):
        x451=self.conv2d136(x450)
        x452=self.batchnorm2d136(x451)
        x453=self.relu133(x452)
        x454=self.conv2d137(x453)
        x455=self.batchnorm2d137(x454)
        x456=self.relu133(x455)
        x457=self.conv2d138(x456)
        return x457

m = M().eval()
x450 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x450)
end = time.time()
print(end-start)
