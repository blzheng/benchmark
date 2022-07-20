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
        self.conv2d69 = Conv2d(120, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x211):
        x212=self.conv2d69(x211)
        x213=self.batchnorm2d41(x212)
        return x213

m = M().eval()
x211 = torch.randn(torch.Size([1, 120, 14, 14]))
start = time.time()
output = m(x211)
end = time.time()
print(end-start)
