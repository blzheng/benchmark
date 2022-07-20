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
        self.conv2d26 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x78, x74):
        x79=x78.sigmoid()
        x80=operator.mul(x74, x79)
        x81=self.conv2d26(x80)
        x82=self.batchnorm2d14(x81)
        return x82

m = M().eval()
x78 = torch.randn(torch.Size([1, 288, 1, 1]))
x74 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x78, x74)
end = time.time()
print(end-start)
