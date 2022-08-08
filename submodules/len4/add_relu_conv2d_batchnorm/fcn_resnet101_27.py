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
        self.relu82 = ReLU(inplace=True)
        self.conv2d88 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d88 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x290, x282):
        x291=operator.add(x290, x282)
        x292=self.relu82(x291)
        x293=self.conv2d88(x292)
        x294=self.batchnorm2d88(x293)
        return x294

m = M().eval()
x290 = torch.randn(torch.Size([1, 1024, 28, 28]))
x282 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x290, x282)
end = time.time()
print(end-start)
