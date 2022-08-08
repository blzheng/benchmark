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
        self.conv2d23 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x67, x63, x57):
        x68=x67.sigmoid()
        x69=operator.mul(x63, x68)
        x70=self.conv2d23(x69)
        x71=self.batchnorm2d13(x70)
        x72=operator.add(x71, x57)
        x73=self.conv2d24(x72)
        x74=self.batchnorm2d14(x73)
        return x74

m = M().eval()
x67 = torch.randn(torch.Size([1, 192, 1, 1]))
x63 = torch.randn(torch.Size([1, 192, 56, 56]))
x57 = torch.randn(torch.Size([1, 32, 56, 56]))
start = time.time()
output = m(x67, x63, x57)
end = time.time()
print(end-start)
