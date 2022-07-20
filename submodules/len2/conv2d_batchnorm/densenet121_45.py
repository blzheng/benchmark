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
        self.conv2d92 = Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d93 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x330):
        x331=self.conv2d92(x330)
        x332=self.batchnorm2d93(x331)
        return x332

m = M().eval()
x330 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x330)
end = time.time()
print(end-start)
