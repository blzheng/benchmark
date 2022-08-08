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
        self.conv2d32 = Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x93):
        x94=self.conv2d32(x93)
        x95=self.batchnorm2d26(x94)
        return x95

m = M().eval()
x93 = torch.randn(torch.Size([1, 184, 14, 14]))
start = time.time()
output = m(x93)
end = time.time()
print(end-start)
