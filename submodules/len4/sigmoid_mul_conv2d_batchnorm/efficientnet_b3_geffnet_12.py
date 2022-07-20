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
        self.conv2d63 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x185, x181):
        x186=x185.sigmoid()
        x187=operator.mul(x181, x186)
        x188=self.conv2d63(x187)
        x189=self.batchnorm2d37(x188)
        return x189

m = M().eval()
x185 = torch.randn(torch.Size([1, 576, 1, 1]))
x181 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x185, x181)
end = time.time()
print(end-start)
