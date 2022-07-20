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
        self.conv2d42 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x130):
        x131=self.conv2d42(x130)
        x132=self.batchnorm2d24(x131)
        return x132

m = M().eval()
x130 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x130)
end = time.time()
print(end-start)
