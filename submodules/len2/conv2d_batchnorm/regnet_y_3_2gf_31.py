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
        self.conv2d49 = Conv2d(576, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x153):
        x154=self.conv2d49(x153)
        x155=self.batchnorm2d31(x154)
        return x155

m = M().eval()
x153 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x153)
end = time.time()
print(end-start)
