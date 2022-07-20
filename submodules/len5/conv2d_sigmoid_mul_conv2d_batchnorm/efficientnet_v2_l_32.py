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
        self.conv2d196 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid32 = Sigmoid()
        self.conv2d197 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d131 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x632, x629):
        x633=self.conv2d196(x632)
        x634=self.sigmoid32(x633)
        x635=operator.mul(x634, x629)
        x636=self.conv2d197(x635)
        x637=self.batchnorm2d131(x636)
        return x637

m = M().eval()
x632 = torch.randn(torch.Size([1, 96, 1, 1]))
x629 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x632, x629)
end = time.time()
print(end-start)
