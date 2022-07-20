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
        self.conv2d97 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d57 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x302, x287):
        x303=operator.add(x302, x287)
        x304=self.conv2d97(x303)
        x305=self.batchnorm2d57(x304)
        return x305

m = M().eval()
x302 = torch.randn(torch.Size([1, 160, 14, 14]))
x287 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x302, x287)
end = time.time()
print(end-start)
