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
        self.relu22 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)

    def forward(self, x89):
        x90=self.relu22(x89)
        x91=self.conv2d28(x90)
        x92=self.batchnorm2d28(x91)
        x93=self.relu25(x92)
        return x93

m = M().eval()
x89 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x89)
end = time.time()
print(end-start)
