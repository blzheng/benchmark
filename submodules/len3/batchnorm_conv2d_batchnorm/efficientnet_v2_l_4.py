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
        self.batchnorm2d122 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d183 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d123 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x590):
        x591=self.batchnorm2d122(x590)
        x592=self.conv2d183(x591)
        x593=self.batchnorm2d123(x592)
        return x593

m = M().eval()
x590 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x590)
end = time.time()
print(end-start)
