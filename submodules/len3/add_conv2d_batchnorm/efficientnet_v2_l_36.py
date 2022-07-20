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
        self.conv2d148 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d102 = BatchNorm2d(1344, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x480, x465):
        x481=operator.add(x480, x465)
        x482=self.conv2d148(x481)
        x483=self.batchnorm2d102(x482)
        return x483

m = M().eval()
x480 = torch.randn(torch.Size([1, 224, 14, 14]))
x465 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x480, x465)
end = time.time()
print(end-start)
