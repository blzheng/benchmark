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
        self.conv2d67 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d68 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(176, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x217, x214):
        x218=self.conv2d67(x217)
        x219=self.sigmoid8(x218)
        x220=operator.mul(x219, x214)
        x221=self.conv2d68(x220)
        x222=self.batchnorm2d50(x221)
        return x222

m = M().eval()
x217 = torch.randn(torch.Size([1, 44, 1, 1]))
x214 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x217, x214)
end = time.time()
print(end-start)
