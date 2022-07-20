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
        self.batchnorm2d43 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x216, x203):
        x217=self.batchnorm2d43(x216)
        x218=operator.add(x203, x217)
        x219=self.relu52(x218)
        x220=self.conv2d70(x219)
        x221=self.batchnorm2d44(x220)
        return x221

m = M().eval()
x216 = torch.randn(torch.Size([1, 440, 7, 7]))
x203 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x216, x203)
end = time.time()
print(end-start)
