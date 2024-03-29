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
        self.conv2d40 = Conv2d(864, 864, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=864, bias=False)
        self.batchnorm2d40 = BatchNorm2d(864, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(144, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x114):
        x115=self.conv2d40(x114)
        x116=self.batchnorm2d40(x115)
        x117=self.relu27(x116)
        x118=self.conv2d41(x117)
        x119=self.batchnorm2d41(x118)
        return x119

m = M().eval()
x114 = torch.randn(torch.Size([1, 864, 7, 7]))
start = time.time()
output = m(x114)
end = time.time()
print(end-start)
