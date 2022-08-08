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
        self.conv2d37 = Conv2d(112, 1232, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d38 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x114, x111, x105):
        x115=self.conv2d37(x114)
        x116=self.sigmoid6(x115)
        x117=operator.mul(x116, x111)
        x118=self.conv2d38(x117)
        x119=self.batchnorm2d24(x118)
        x120=operator.add(x105, x119)
        x121=self.relu28(x120)
        x122=self.conv2d39(x121)
        x123=self.batchnorm2d25(x122)
        return x123

m = M().eval()
x114 = torch.randn(torch.Size([1, 112, 1, 1]))
x111 = torch.randn(torch.Size([1, 1232, 14, 14]))
x105 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x114, x111, x105)
end = time.time()
print(end-start)
