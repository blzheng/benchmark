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
        self.conv2d105 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d105 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x346, x340):
        x347=self.conv2d105(x346)
        x348=self.batchnorm2d105(x347)
        x349=operator.add(x348, x340)
        return x349

m = M().eval()
x346 = torch.randn(torch.Size([1, 256, 14, 14]))
x340 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x346, x340)
end = time.time()
print(end-start)
