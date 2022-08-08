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
        self.conv2d94 = Conv2d(576, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu73 = ReLU(inplace=True)
        self.conv2d95 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
        self.batchnorm2d59 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu74 = ReLU(inplace=True)

    def forward(self, x297):
        x298=self.conv2d94(x297)
        x299=self.batchnorm2d58(x298)
        x300=self.relu73(x299)
        x301=self.conv2d95(x300)
        x302=self.batchnorm2d59(x301)
        x303=self.relu74(x302)
        return x303

m = M().eval()
x297 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x297)
end = time.time()
print(end-start)
