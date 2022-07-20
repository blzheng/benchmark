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
        self.conv2d49 = Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x150, x135):
        x151=operator.add(x150, x135)
        x152=self.conv2d49(x151)
        x153=self.batchnorm2d29(x152)
        return x153

m = M().eval()
x150 = torch.randn(torch.Size([1, 56, 28, 28]))
x135 = torch.randn(torch.Size([1, 56, 28, 28]))
start = time.time()
output = m(x150, x135)
end = time.time()
print(end-start)
