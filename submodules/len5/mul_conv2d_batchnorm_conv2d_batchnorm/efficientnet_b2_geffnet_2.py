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
        self.conv2d43 = Conv2d(288, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d44 = Conv2d(88, 528, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x122, x127):
        x128=operator.mul(x122, x127)
        x129=self.conv2d43(x128)
        x130=self.batchnorm2d25(x129)
        x131=self.conv2d44(x130)
        x132=self.batchnorm2d26(x131)
        return x132

m = M().eval()
x122 = torch.randn(torch.Size([1, 288, 14, 14]))
x127 = torch.randn(torch.Size([1, 288, 1, 1]))
start = time.time()
output = m(x122, x127)
end = time.time()
print(end-start)
