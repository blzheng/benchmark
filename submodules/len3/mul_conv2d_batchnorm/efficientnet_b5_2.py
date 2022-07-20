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
        self.conv2d12 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x35, x30):
        x36=operator.mul(x35, x30)
        x37=self.conv2d12(x36)
        x38=self.batchnorm2d6(x37)
        return x38

m = M().eval()
x35 = torch.randn(torch.Size([1, 24, 1, 1]))
x30 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x35, x30)
end = time.time()
print(end-start)
