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
        self.conv2d0 = Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d0 = BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x0):
        x3=self.conv2d0(x0)
        x4=self.batchnorm2d0(x3)
        return x4

m = M().eval()
x0 = torch.randn(torch.Size([1, 3, 224, 224]))
start = time.time()
output = m(x0)
end = time.time()
print(end-start)