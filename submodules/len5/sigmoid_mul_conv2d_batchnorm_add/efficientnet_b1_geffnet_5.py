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
        self.conv2d108 = Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x318, x314, x336):
        x319=x318.sigmoid()
        x320=operator.mul(x314, x319)
        x321=self.conv2d108(x320)
        x322=self.batchnorm2d64(x321)
        x337=operator.add(x336, x322)
        return x337

m = M().eval()
x318 = torch.randn(torch.Size([1, 1152, 1, 1]))
x314 = torch.randn(torch.Size([1, 1152, 7, 7]))
x336 = torch.randn(torch.Size([1, 320, 7, 7]))
start = time.time()
output = m(x318, x314, x336)
end = time.time()
print(end-start)
