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
        self.relu0 = ReLU(inplace=True)
        self.conv2d1 = Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d1 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)

    def forward(self, x2):
        x3=self.relu0(x2)
        x4=self.conv2d1(x3)
        x5=self.batchnorm2d1(x4)
        x6=self.relu1(x5)
        return x6

m = M().eval()
x2 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x2)
end = time.time()
print(end-start)
