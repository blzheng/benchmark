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
        self.conv2d217 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid38 = Sigmoid()
        self.conv2d218 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d140 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x695, x692):
        x696=self.conv2d217(x695)
        x697=self.sigmoid38(x696)
        x698=operator.mul(x697, x692)
        x699=self.conv2d218(x698)
        x700=self.batchnorm2d140(x699)
        return x700

m = M().eval()
x695 = torch.randn(torch.Size([1, 76, 1, 1]))
x692 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x695, x692)
end = time.time()
print(end-start)
