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
        self.sigmoid1 = Sigmoid()
        self.conv2d8 = Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x21, x17):
        x22=self.sigmoid1(x21)
        x23=operator.mul(x22, x17)
        x24=self.conv2d8(x23)
        x25=self.batchnorm2d4(x24)
        return x25

m = M().eval()
x21 = torch.randn(torch.Size([1, 16, 1, 1]))
x17 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x21, x17)
end = time.time()
print(end-start)
