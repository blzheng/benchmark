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
        self.conv2d167 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid28 = Sigmoid()
        self.conv2d168 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d110 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x535, x532):
        x536=self.conv2d167(x535)
        x537=self.sigmoid28(x536)
        x538=operator.mul(x537, x532)
        x539=self.conv2d168(x538)
        x540=self.batchnorm2d110(x539)
        return x540

m = M().eval()
x535 = torch.randn(torch.Size([1, 76, 1, 1]))
x532 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x535, x532)
end = time.time()
print(end-start)
