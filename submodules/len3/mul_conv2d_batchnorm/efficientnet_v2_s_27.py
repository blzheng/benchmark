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
        self.conv2d158 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d102 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x502, x497):
        x503=operator.mul(x502, x497)
        x504=self.conv2d158(x503)
        x505=self.batchnorm2d102(x504)
        return x505

m = M().eval()
x502 = torch.randn(torch.Size([1, 1536, 1, 1]))
x497 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x502, x497)
end = time.time()
print(end-start)
