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
        self.conv2d49 = Conv2d(1488, 1488, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1488, bias=False)
        self.batchnorm2d49 = BatchNorm2d(1488, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x141):
        x142=self.conv2d49(x141)
        x143=self.batchnorm2d49(x142)
        return x143

m = M().eval()
x141 = torch.randn(torch.Size([1, 1488, 7, 7]))
start = time.time()
output = m(x141)
end = time.time()
print(end-start)
