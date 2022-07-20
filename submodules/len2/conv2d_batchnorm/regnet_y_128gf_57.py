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
        self.conv2d93 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x293):
        x294=self.conv2d93(x293)
        x295=self.batchnorm2d57(x294)
        return x295

m = M().eval()
x293 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x293)
end = time.time()
print(end-start)
