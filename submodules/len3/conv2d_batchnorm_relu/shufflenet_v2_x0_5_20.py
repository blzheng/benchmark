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
        self.conv2d31 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)

    def forward(self, x192):
        x193=self.conv2d31(x192)
        x194=self.batchnorm2d31(x193)
        x195=self.relu20(x194)
        return x195

m = M().eval()
x192 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x192)
end = time.time()
print(end-start)
