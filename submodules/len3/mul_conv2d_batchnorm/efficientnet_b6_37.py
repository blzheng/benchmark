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
        self.conv2d187 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d111 = BatchNorm2d(344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x585, x580):
        x586=operator.mul(x585, x580)
        x587=self.conv2d187(x586)
        x588=self.batchnorm2d111(x587)
        return x588

m = M().eval()
x585 = torch.randn(torch.Size([1, 2064, 1, 1]))
x580 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x585, x580)
end = time.time()
print(end-start)
