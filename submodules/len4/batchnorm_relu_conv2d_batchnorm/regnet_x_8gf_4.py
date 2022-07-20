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
        self.batchnorm2d6 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x19):
        x20=self.batchnorm2d6(x19)
        x21=self.relu5(x20)
        x22=self.conv2d7(x21)
        x23=self.batchnorm2d7(x22)
        return x23

m = M().eval()
x19 = torch.randn(torch.Size([1, 80, 56, 56]))
start = time.time()
output = m(x19)
end = time.time()
print(end-start)
