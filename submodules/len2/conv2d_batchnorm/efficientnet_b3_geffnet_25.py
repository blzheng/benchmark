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
        self.conv2d43 = Conv2d(288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x128):
        x129=self.conv2d43(x128)
        x130=self.batchnorm2d25(x129)
        return x130

m = M().eval()
x128 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x128)
end = time.time()
print(end-start)
