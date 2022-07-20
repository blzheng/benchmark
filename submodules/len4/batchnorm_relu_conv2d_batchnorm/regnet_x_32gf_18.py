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
        self.batchnorm2d28 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x91):
        x92=self.batchnorm2d28(x91)
        x93=self.relu26(x92)
        x94=self.conv2d29(x93)
        x95=self.batchnorm2d29(x94)
        return x95

m = M().eval()
x91 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x91)
end = time.time()
print(end-start)
