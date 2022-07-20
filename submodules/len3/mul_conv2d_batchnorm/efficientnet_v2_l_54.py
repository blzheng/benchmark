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
        self.conv2d307 = Conv2d(2304, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d197 = BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x986, x981):
        x987=operator.mul(x986, x981)
        x988=self.conv2d307(x987)
        x989=self.batchnorm2d197(x988)
        return x989

m = M().eval()
x986 = torch.randn(torch.Size([1, 2304, 1, 1]))
x981 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x986, x981)
end = time.time()
print(end-start)
