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
        self.sigmoid3 = Sigmoid()
        self.conv2d19 = Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d20 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x54, x50):
        x55=self.sigmoid3(x54)
        x56=operator.mul(x55, x50)
        x57=self.conv2d19(x56)
        x58=self.batchnorm2d11(x57)
        x59=self.conv2d20(x58)
        x60=self.batchnorm2d12(x59)
        return x60

m = M().eval()
x54 = torch.randn(torch.Size([1, 144, 1, 1]))
x50 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x54, x50)
end = time.time()
print(end-start)
