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
        self.conv2d27 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x85, x79):
        x86=self.conv2d27(x85)
        x87=self.batchnorm2d27(x86)
        x88=operator.add(x79, x87)
        return x88

m = M().eval()
x85 = torch.randn(torch.Size([1, 408, 14, 14]))
x79 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x85, x79)
end = time.time()
print(end-start)
