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
        self.conv2d123 = Conv2d(1392, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d129 = Conv2d(384, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x363, x359, x381):
        x364=x363.sigmoid()
        x365=operator.mul(x359, x364)
        x366=self.conv2d123(x365)
        x367=self.batchnorm2d73(x366)
        x382=operator.add(x381, x367)
        x383=self.conv2d129(x382)
        return x383

m = M().eval()
x363 = torch.randn(torch.Size([1, 1392, 1, 1]))
x359 = torch.randn(torch.Size([1, 1392, 7, 7]))
x381 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x363, x359, x381)
end = time.time()
print(end-start)
