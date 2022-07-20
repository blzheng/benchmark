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
        self.conv2d266 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d158 = BatchNorm2d(640, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x836, x831):
        x837=operator.mul(x836, x831)
        x838=self.conv2d266(x837)
        x839=self.batchnorm2d158(x838)
        return x839

m = M().eval()
x836 = torch.randn(torch.Size([1, 3840, 1, 1]))
x831 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x836, x831)
end = time.time()
print(end-start)
