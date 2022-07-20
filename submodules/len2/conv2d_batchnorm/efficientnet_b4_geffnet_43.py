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
        self.conv2d73 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x217):
        x218=self.conv2d73(x217)
        x219=self.batchnorm2d43(x218)
        return x219

m = M().eval()
x217 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x217)
end = time.time()
print(end-start)
