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
        self.conv2d78 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x239):
        x240=self.conv2d78(x239)
        x241=self.batchnorm2d46(x240)
        return x241

m = M().eval()
x239 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x239)
end = time.time()
print(end-start)
