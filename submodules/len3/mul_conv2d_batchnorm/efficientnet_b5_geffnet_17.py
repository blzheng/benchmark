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
        self.conv2d87 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x254, x259):
        x260=operator.mul(x254, x259)
        x261=self.conv2d87(x260)
        x262=self.batchnorm2d51(x261)
        return x262

m = M().eval()
x254 = torch.randn(torch.Size([1, 768, 14, 14]))
x259 = torch.randn(torch.Size([1, 768, 1, 1]))
start = time.time()
output = m(x254, x259)
end = time.time()
print(end-start)
