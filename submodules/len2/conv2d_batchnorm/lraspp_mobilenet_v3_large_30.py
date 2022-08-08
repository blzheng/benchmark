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
        self.conv2d36 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x105):
        x106=self.conv2d36(x105)
        x107=self.batchnorm2d30(x106)
        return x107

m = M().eval()
x105 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x105)
end = time.time()
print(end-start)
