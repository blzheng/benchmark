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
        self.conv2d37 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x106, x111):
        x112=operator.mul(x106, x111)
        x113=self.conv2d37(x112)
        x114=self.batchnorm2d21(x113)
        return x114

m = M().eval()
x106 = torch.randn(torch.Size([1, 240, 56, 56]))
x111 = torch.randn(torch.Size([1, 240, 1, 1]))
start = time.time()
output = m(x106, x111)
end = time.time()
print(end-start)
