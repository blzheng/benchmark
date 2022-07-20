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
        self.batchnorm2d5 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(232, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)

    def forward(self, x22):
        x23=self.batchnorm2d5(x22)
        x24=self.relu5(x23)
        x25=self.conv2d8(x24)
        x26=self.batchnorm2d6(x25)
        x27=self.relu6(x26)
        return x27

m = M().eval()
x22 = torch.randn(torch.Size([1, 232, 56, 56]))
start = time.time()
output = m(x22)
end = time.time()
print(end-start)
