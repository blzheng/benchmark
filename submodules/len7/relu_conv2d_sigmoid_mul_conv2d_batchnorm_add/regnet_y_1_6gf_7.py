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
        self.relu31 = ReLU()
        self.conv2d41 = Conv2d(30, 120, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()
        self.conv2d42 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x127, x125, x119):
        x128=self.relu31(x127)
        x129=self.conv2d41(x128)
        x130=self.sigmoid7(x129)
        x131=operator.mul(x130, x125)
        x132=self.conv2d42(x131)
        x133=self.batchnorm2d26(x132)
        x134=operator.add(x119, x133)
        return x134

m = M().eval()
x127 = torch.randn(torch.Size([1, 30, 1, 1]))
x125 = torch.randn(torch.Size([1, 120, 28, 28]))
x119 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x127, x125, x119)
end = time.time()
print(end-start)
