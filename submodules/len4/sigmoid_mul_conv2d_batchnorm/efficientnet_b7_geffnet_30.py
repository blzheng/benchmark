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
        self.conv2d151 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d89 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x450, x446):
        x451=x450.sigmoid()
        x452=operator.mul(x446, x451)
        x453=self.conv2d151(x452)
        x454=self.batchnorm2d89(x453)
        return x454

m = M().eval()
x450 = torch.randn(torch.Size([1, 1344, 1, 1]))
x446 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x450, x446)
end = time.time()
print(end-start)
