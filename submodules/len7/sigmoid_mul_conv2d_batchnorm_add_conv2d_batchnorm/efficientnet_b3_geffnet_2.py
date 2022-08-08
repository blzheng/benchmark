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
        self.conv2d63 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d64 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x185, x181, x175):
        x186=x185.sigmoid()
        x187=operator.mul(x181, x186)
        x188=self.conv2d63(x187)
        x189=self.batchnorm2d37(x188)
        x190=operator.add(x189, x175)
        x191=self.conv2d64(x190)
        x192=self.batchnorm2d38(x191)
        return x192

m = M().eval()
x185 = torch.randn(torch.Size([1, 576, 1, 1]))
x181 = torch.randn(torch.Size([1, 576, 14, 14]))
x175 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x185, x181, x175)
end = time.time()
print(end-start)
