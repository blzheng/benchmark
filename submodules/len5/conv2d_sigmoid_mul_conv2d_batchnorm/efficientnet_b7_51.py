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
        self.conv2d255 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid51 = Sigmoid()
        self.conv2d256 = Conv2d(2304, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d152 = BatchNorm2d(640, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x804, x801):
        x805=self.conv2d255(x804)
        x806=self.sigmoid51(x805)
        x807=operator.mul(x806, x801)
        x808=self.conv2d256(x807)
        x809=self.batchnorm2d152(x808)
        return x809

m = M().eval()
x804 = torch.randn(torch.Size([1, 96, 1, 1]))
x801 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x804, x801)
end = time.time()
print(end-start)
