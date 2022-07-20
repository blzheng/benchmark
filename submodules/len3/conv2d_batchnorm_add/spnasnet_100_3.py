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
        self.conv2d47 = Conv2d(288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x153, x146):
        x154=self.conv2d47(x153)
        x155=self.batchnorm2d47(x154)
        x156=operator.add(x155, x146)
        return x156

m = M().eval()
x153 = torch.randn(torch.Size([1, 288, 14, 14]))
x146 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x153, x146)
end = time.time()
print(end-start)
