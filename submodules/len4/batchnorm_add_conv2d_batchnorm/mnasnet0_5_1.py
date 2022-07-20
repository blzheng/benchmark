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
        self.batchnorm2d20 = BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d21 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(144, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x58, x51):
        x59=self.batchnorm2d20(x58)
        x60=operator.add(x59, x51)
        x61=self.conv2d21(x60)
        x62=self.batchnorm2d21(x61)
        return x62

m = M().eval()
x58 = torch.randn(torch.Size([1, 24, 28, 28]))
x51 = torch.randn(torch.Size([1, 24, 28, 28]))
start = time.time()
output = m(x58, x51)
end = time.time()
print(end-start)
