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
        self.conv2d8 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x23, x17):
        x24=self.conv2d8(x23)
        x25=self.batchnorm2d8(x24)
        x26=operator.add(x17, x25)
        return x26

m = M().eval()
x23 = torch.randn(torch.Size([1, 64, 28, 28]))
x17 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x23, x17)
end = time.time()
print(end-start)
