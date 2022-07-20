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
        self.conv2d188 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d122 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x601, x596):
        x602=operator.mul(x601, x596)
        x603=self.conv2d188(x602)
        x604=self.batchnorm2d122(x603)
        return x604

m = M().eval()
x601 = torch.randn(torch.Size([1, 1824, 1, 1]))
x596 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x601, x596)
end = time.time()
print(end-start)
