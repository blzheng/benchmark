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
        self.sigmoid19 = Sigmoid()
        self.conv2d98 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x301, x297):
        x302=self.sigmoid19(x301)
        x303=operator.mul(x302, x297)
        x304=self.conv2d98(x303)
        x305=self.batchnorm2d58(x304)
        return x305

m = M().eval()
x301 = torch.randn(torch.Size([1, 960, 1, 1]))
x297 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x301, x297)
end = time.time()
print(end-start)
