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
        self.conv2d109 = Conv2d(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x346, x331):
        x347=operator.add(x346, x331)
        x348=self.conv2d109(x347)
        x349=self.batchnorm2d73(x348)
        return x349

m = M().eval()
x346 = torch.randn(torch.Size([1, 256, 7, 7]))
x331 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x346, x331)
end = time.time()
print(end-start)
