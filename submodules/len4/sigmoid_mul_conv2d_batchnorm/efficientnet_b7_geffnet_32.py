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
        self.conv2d161 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d95 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x480, x476):
        x481=x480.sigmoid()
        x482=operator.mul(x476, x481)
        x483=self.conv2d161(x482)
        x484=self.batchnorm2d95(x483)
        return x484

m = M().eval()
x480 = torch.randn(torch.Size([1, 1344, 1, 1]))
x476 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x480, x476)
end = time.time()
print(end-start)
