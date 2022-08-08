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
        self.conv2d58 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu55 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(896, 896, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=7, bias=False)
        self.batchnorm2d59 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU(inplace=True)
        self.conv2d60 = Conv2d(896, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x189):
        x190=self.conv2d58(x189)
        x191=self.batchnorm2d58(x190)
        x192=self.relu55(x191)
        x193=self.conv2d59(x192)
        x194=self.batchnorm2d59(x193)
        x195=self.relu56(x194)
        x196=self.conv2d60(x195)
        x197=self.batchnorm2d60(x196)
        return x197

m = M().eval()
x189 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x189)
end = time.time()
print(end-start)
