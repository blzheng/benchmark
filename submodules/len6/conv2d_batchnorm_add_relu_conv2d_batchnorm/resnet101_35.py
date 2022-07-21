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
        self.conv2d100 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d100 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)
        self.conv2d101 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x328, x322):
        x329=self.conv2d100(x328)
        x330=self.batchnorm2d100(x329)
        x331=operator.add(x330, x322)
        x332=self.relu94(x331)
        x333=self.conv2d101(x332)
        x334=self.batchnorm2d101(x333)
        return x334

m = M().eval()
x328 = torch.randn(torch.Size([1, 512, 7, 7]))
x322 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x328, x322)
end = time.time()
print(end-start)
