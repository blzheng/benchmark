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
        self.batchnorm2d28 = BatchNorm2d(240, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x81):
        x82=self.batchnorm2d28(x81)
        x83=self.relu19(x82)
        x84=self.conv2d29(x83)
        return x84

m = M().eval()
x81 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x81)
end = time.time()
print(end-start)
