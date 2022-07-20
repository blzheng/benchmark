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
        self.conv2d27 = Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d28 = Conv2d(64, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x88):
        x89=self.conv2d27(x88)
        x90=self.batchnorm2d27(x89)
        x91=self.conv2d28(x90)
        return x91

m = M().eval()
x88 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x88)
end = time.time()
print(end-start)
