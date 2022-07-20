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
        self.conv2d43 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x138, x129):
        x139=operator.add(x138, x129)
        x140=self.conv2d43(x139)
        x141=self.batchnorm2d43(x140)
        return x141

m = M().eval()
x138 = torch.randn(torch.Size([1, 112, 14, 14]))
x129 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x138, x129)
end = time.time()
print(end-start)
