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
        self.conv2d29 = Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d30 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x79, x84):
        x85=operator.mul(x79, x84)
        x86=self.conv2d29(x85)
        x87=self.batchnorm2d17(x86)
        x88=self.conv2d30(x87)
        return x88

m = M().eval()
x79 = torch.randn(torch.Size([1, 240, 14, 14]))
x84 = torch.randn(torch.Size([1, 240, 1, 1]))
start = time.time()
output = m(x79, x84)
end = time.time()
print(end-start)