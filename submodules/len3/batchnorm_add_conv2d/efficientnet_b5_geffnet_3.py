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
        self.batchnorm2d78 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d133 = Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x395, x382):
        x396=self.batchnorm2d78(x395)
        x397=operator.add(x396, x382)
        x398=self.conv2d133(x397)
        return x398

m = M().eval()
x395 = torch.randn(torch.Size([1, 176, 14, 14]))
x382 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x395, x382)
end = time.time()
print(end-start)
