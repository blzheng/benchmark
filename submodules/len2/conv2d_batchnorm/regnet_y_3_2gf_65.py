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
        self.conv2d105 = Conv2d(576, 1512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(1512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x329):
        x332=self.conv2d105(x329)
        x333=self.batchnorm2d65(x332)
        return x333

m = M().eval()
x329 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x329)
end = time.time()
print(end-start)
