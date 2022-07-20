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
        self.conv2d117 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x347, x343):
        x348=x347.sigmoid()
        x349=operator.mul(x343, x348)
        x350=self.conv2d117(x349)
        x351=self.batchnorm2d69(x350)
        return x351

m = M().eval()
x347 = torch.randn(torch.Size([1, 1056, 1, 1]))
x343 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x347, x343)
end = time.time()
print(end-start)
