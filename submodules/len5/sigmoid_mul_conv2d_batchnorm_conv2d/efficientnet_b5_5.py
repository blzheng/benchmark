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
        self.sigmoid27 = Sigmoid()
        self.conv2d137 = Conv2d(1056, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d81 = BatchNorm2d(304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d138 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x426, x422):
        x427=self.sigmoid27(x426)
        x428=operator.mul(x427, x422)
        x429=self.conv2d137(x428)
        x430=self.batchnorm2d81(x429)
        x431=self.conv2d138(x430)
        return x431

m = M().eval()
x426 = torch.randn(torch.Size([1, 1056, 1, 1]))
x422 = torch.randn(torch.Size([1, 1056, 7, 7]))
start = time.time()
output = m(x426, x422)
end = time.time()
print(end-start)
