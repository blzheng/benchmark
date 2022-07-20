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
        self.conv2d15 = Conv2d(96, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x43):
        x44=self.conv2d15(x43)
        x45=self.batchnorm2d11(x44)
        return x45

m = M().eval()
x43 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x43)
end = time.time()
print(end-start)
