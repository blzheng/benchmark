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
        self.conv2d241 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d143 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x715, x720):
        x721=operator.mul(x715, x720)
        x722=self.conv2d241(x721)
        x723=self.batchnorm2d143(x722)
        return x723

m = M().eval()
x715 = torch.randn(torch.Size([1, 2304, 7, 7]))
x720 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x715, x720)
end = time.time()
print(end-start)
