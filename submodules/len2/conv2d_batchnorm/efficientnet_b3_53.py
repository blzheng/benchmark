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
        self.conv2d89 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x275):
        x276=self.conv2d89(x275)
        x277=self.batchnorm2d53(x276)
        return x277

m = M().eval()
x275 = torch.randn(torch.Size([1, 136, 14, 14]))
start = time.time()
output = m(x275)
end = time.time()
print(end-start)
