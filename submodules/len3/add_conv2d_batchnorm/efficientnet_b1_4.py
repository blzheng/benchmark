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
        self.conv2d39 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x118, x103):
        x119=operator.add(x118, x103)
        x120=self.conv2d39(x119)
        x121=self.batchnorm2d23(x120)
        return x121

m = M().eval()
x118 = torch.randn(torch.Size([1, 40, 28, 28]))
x103 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x118, x103)
end = time.time()
print(end-start)
