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
        self.conv2d36 = Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)
        self.conv2d37 = Conv2d(288, 672, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x115, x109):
        x116=self.conv2d36(x115)
        x117=self.batchnorm2d36(x116)
        x118=operator.add(x109, x117)
        x119=self.relu33(x118)
        x120=self.conv2d37(x119)
        return x120

m = M().eval()
x115 = torch.randn(torch.Size([1, 288, 14, 14]))
x109 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x115, x109)
end = time.time()
print(end-start)
