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
        self.conv2d109 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x322, x308):
        x323=operator.add(x322, x308)
        x324=self.conv2d109(x323)
        x325=self.batchnorm2d65(x324)
        return x325

m = M().eval()
x322 = torch.randn(torch.Size([1, 232, 7, 7]))
x308 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x322, x308)
end = time.time()
print(end-start)
