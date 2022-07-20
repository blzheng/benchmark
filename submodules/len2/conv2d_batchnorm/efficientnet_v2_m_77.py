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
        self.conv2d113 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d77 = BatchNorm2d(176, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x364):
        x365=self.conv2d113(x364)
        x366=self.batchnorm2d77(x365)
        return x366

m = M().eval()
x364 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x364)
end = time.time()
print(end-start)
