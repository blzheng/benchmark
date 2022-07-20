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
        self.conv2d42 = Conv2d(54, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()
        self.conv2d43 = Conv2d(576, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x130, x127):
        x131=self.conv2d42(x130)
        x132=self.sigmoid7(x131)
        x133=operator.mul(x132, x127)
        x134=self.conv2d43(x133)
        x135=self.batchnorm2d27(x134)
        return x135

m = M().eval()
x130 = torch.randn(torch.Size([1, 54, 1, 1]))
x127 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x130, x127)
end = time.time()
print(end-start)
