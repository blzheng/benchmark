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
        self.conv2d30 = Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x86, x78):
        x87=operator.add(x86, x78)
        x88=self.conv2d30(x87)
        x89=self.batchnorm2d24(x88)
        return x89

m = M().eval()
x86 = torch.randn(torch.Size([1, 80, 14, 14]))
x78 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x86, x78)
end = time.time()
print(end-start)
