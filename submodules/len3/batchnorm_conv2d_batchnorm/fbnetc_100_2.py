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
        self.batchnorm2d27 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d28 = Conv2d(64, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x89):
        x90=self.batchnorm2d27(x89)
        x91=self.conv2d28(x90)
        x92=self.batchnorm2d28(x91)
        return x92

m = M().eval()
x89 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x89)
end = time.time()
print(end-start)
