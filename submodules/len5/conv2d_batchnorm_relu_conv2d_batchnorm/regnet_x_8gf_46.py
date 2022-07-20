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
        self.conv2d72 = Conv2d(1920, 1920, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
        self.batchnorm2d72 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu68 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(1920, 1920, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x234):
        x235=self.conv2d72(x234)
        x236=self.batchnorm2d72(x235)
        x237=self.relu68(x236)
        x238=self.conv2d73(x237)
        x239=self.batchnorm2d73(x238)
        return x239

m = M().eval()
x234 = torch.randn(torch.Size([1, 1920, 14, 14]))
start = time.time()
output = m(x234)
end = time.time()
print(end-start)
