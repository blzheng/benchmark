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
        self.conv2d50 = Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x146):
        x147=self.conv2d50(x146)
        x148=self.batchnorm2d38(x147)
        return x148

m = M().eval()
x146 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x146)
end = time.time()
print(end-start)
