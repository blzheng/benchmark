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
        self.sigmoid25 = Sigmoid()
        self.conv2d126 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d74 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x393, x389):
        x394=self.sigmoid25(x393)
        x395=operator.mul(x394, x389)
        x396=self.conv2d126(x395)
        x397=self.batchnorm2d74(x396)
        return x397

m = M().eval()
x393 = torch.randn(torch.Size([1, 960, 1, 1]))
x389 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x393, x389)
end = time.time()
print(end-start)
