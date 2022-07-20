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
        self.conv2d133 = Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d79 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x397):
        x398=self.conv2d133(x397)
        x399=self.batchnorm2d79(x398)
        return x399

m = M().eval()
x397 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x397)
end = time.time()
print(end-start)
