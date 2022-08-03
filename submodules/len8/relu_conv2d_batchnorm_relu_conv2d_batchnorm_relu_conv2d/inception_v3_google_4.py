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
        self.conv2d55 = Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
        self.batchnorm2d55 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d56 = Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
        self.batchnorm2d56 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d57 = Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)

    def forward(self, x191):
        x192=torch.nn.functional.relu(x191,inplace=True)
        x193=self.conv2d55(x192)
        x194=self.batchnorm2d55(x193)
        x195=torch.nn.functional.relu(x194,inplace=True)
        x196=self.conv2d56(x195)
        x197=self.batchnorm2d56(x196)
        x198=torch.nn.functional.relu(x197,inplace=True)
        x199=self.conv2d57(x198)
        return x199

m = M().eval()
x191 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x191)
end = time.time()
print(end-start)
