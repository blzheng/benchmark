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
        self.sigmoid40 = Sigmoid()
        self.conv2d202 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d120 = BatchNorm2d(344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x632, x628):
        x633=self.sigmoid40(x632)
        x634=operator.mul(x633, x628)
        x635=self.conv2d202(x634)
        x636=self.batchnorm2d120(x635)
        return x636

m = M().eval()
x632 = torch.randn(torch.Size([1, 2064, 1, 1]))
x628 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x632, x628)
end = time.time()
print(end-start)
