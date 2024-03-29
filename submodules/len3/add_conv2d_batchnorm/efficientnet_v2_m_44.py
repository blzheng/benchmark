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
        self.conv2d209 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d135 = BatchNorm2d(1824, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x669, x654):
        x670=operator.add(x669, x654)
        x671=self.conv2d209(x670)
        x672=self.batchnorm2d135(x671)
        return x672

m = M().eval()
x669 = torch.randn(torch.Size([1, 304, 7, 7]))
x654 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x669, x654)
end = time.time()
print(end-start)
