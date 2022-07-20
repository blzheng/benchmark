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
        self.conv2d118 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x348, x344, x338):
        x349=x348.sigmoid()
        x350=operator.mul(x344, x349)
        x351=self.conv2d118(x350)
        x352=self.batchnorm2d70(x351)
        x353=operator.add(x352, x338)
        return x353

m = M().eval()
x348 = torch.randn(torch.Size([1, 1392, 1, 1]))
x344 = torch.randn(torch.Size([1, 1392, 7, 7]))
x338 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x348, x344, x338)
end = time.time()
print(end-start)
