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
        self.conv2d157 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d93 = BatchNorm2d(304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x490):
        x491=self.conv2d157(x490)
        x492=self.batchnorm2d93(x491)
        return x492

m = M().eval()
x490 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x490)
end = time.time()
print(end-start)
