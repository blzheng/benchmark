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
        self.conv2d34 = Conv2d(448, 896, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x103):
        x106=self.conv2d34(x103)
        x107=self.batchnorm2d22(x106)
        return x107

m = M().eval()
x103 = torch.randn(torch.Size([1, 448, 28, 28]))
start = time.time()
output = m(x103)
end = time.time()
print(end-start)
