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
        self.batchnorm2d9 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(288, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x33):
        x34=self.batchnorm2d9(x33)
        x35=self.relu9(x34)
        x36=self.conv2d9(x35)
        x37=self.batchnorm2d10(x36)
        x38=self.relu10(x37)
        x39=self.conv2d10(x38)
        return x39

m = M().eval()
x33 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x33)
end = time.time()
print(end-start)
