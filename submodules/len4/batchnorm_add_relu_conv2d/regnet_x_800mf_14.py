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
        self.batchnorm2d37 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x120, x129):
        x121=self.batchnorm2d37(x120)
        x130=operator.add(x121, x129)
        x131=self.relu36(x130)
        x132=self.conv2d41(x131)
        return x132

m = M().eval()
x120 = torch.randn(torch.Size([1, 672, 7, 7]))
x129 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x120, x129)
end = time.time()
print(end-start)
