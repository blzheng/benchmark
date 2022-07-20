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
        self.conv2d34 = Conv2d(160, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x113, x98):
        x114=operator.add(x113, x98)
        x115=self.conv2d34(x114)
        x116=self.batchnorm2d30(x115)
        return x116

m = M().eval()
x113 = torch.randn(torch.Size([1, 160, 14, 14]))
x98 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x113, x98)
end = time.time()
print(end-start)
