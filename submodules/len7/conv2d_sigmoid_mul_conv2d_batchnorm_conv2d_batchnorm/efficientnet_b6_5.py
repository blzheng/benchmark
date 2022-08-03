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
        self.conv2d156 = Conv2d(50, 1200, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid31 = Sigmoid()
        self.conv2d157 = Conv2d(1200, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d93 = BatchNorm2d(344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d158 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d94 = BatchNorm2d(2064, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x489, x486):
        x490=self.conv2d156(x489)
        x491=self.sigmoid31(x490)
        x492=operator.mul(x491, x486)
        x493=self.conv2d157(x492)
        x494=self.batchnorm2d93(x493)
        x495=self.conv2d158(x494)
        x496=self.batchnorm2d94(x495)
        return x496

m = M().eval()
x489 = torch.randn(torch.Size([1, 50, 1, 1]))
x486 = torch.randn(torch.Size([1, 1200, 7, 7]))
start = time.time()
output = m(x489, x486)
end = time.time()
print(end-start)
