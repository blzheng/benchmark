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
        self.relu31 = ReLU(inplace=True)
        self.conv2d36 = Conv2d(400, 400, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d36 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)
        self.conv2d37 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x113):
        x114=self.relu31(x113)
        x115=self.conv2d36(x114)
        x116=self.batchnorm2d36(x115)
        x117=self.relu32(x116)
        x118=self.conv2d37(x117)
        x119=self.batchnorm2d37(x118)
        return x119

m = M().eval()
x113 = torch.randn(torch.Size([1, 400, 14, 14]))
start = time.time()
output = m(x113)
end = time.time()
print(end-start)