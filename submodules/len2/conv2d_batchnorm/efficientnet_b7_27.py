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
        self.conv2d47 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x147):
        x148=self.conv2d47(x147)
        x149=self.batchnorm2d27(x148)
        return x149

m = M().eval()
x147 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x147)
end = time.time()
print(end-start)
