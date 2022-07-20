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
        self.conv2d169 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d111 = BatchNorm2d(1824, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x542):
        x543=self.conv2d169(x542)
        x544=self.batchnorm2d111(x543)
        return x544

m = M().eval()
x542 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x542)
end = time.time()
print(end-start)
