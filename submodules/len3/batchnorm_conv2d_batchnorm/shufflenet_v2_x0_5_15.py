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
        self.batchnorm2d44 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d45 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x281):
        x282=self.batchnorm2d44(x281)
        x283=self.conv2d45(x282)
        x284=self.batchnorm2d45(x283)
        return x284

m = M().eval()
x281 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x281)
end = time.time()
print(end-start)
