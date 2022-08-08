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
        self.conv2d103 = Conv2d(348, 3712, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()
        self.conv2d104 = Conv2d(3712, 3712, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(3712, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x324, x321, x315):
        x325=self.conv2d103(x324)
        x326=self.sigmoid19(x325)
        x327=operator.mul(x326, x321)
        x328=self.conv2d104(x327)
        x329=self.batchnorm2d64(x328)
        x330=operator.add(x315, x329)
        return x330

m = M().eval()
x324 = torch.randn(torch.Size([1, 348, 1, 1]))
x321 = torch.randn(torch.Size([1, 3712, 7, 7]))
x315 = torch.randn(torch.Size([1, 3712, 7, 7]))
start = time.time()
output = m(x324, x321, x315)
end = time.time()
print(end-start)
