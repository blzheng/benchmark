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
        self.conv2d121 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()
        self.conv2d122 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d86 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x394, x391):
        x395=self.conv2d121(x394)
        x396=self.sigmoid17(x395)
        x397=operator.mul(x396, x391)
        x398=self.conv2d122(x397)
        x399=self.batchnorm2d86(x398)
        return x399

m = M().eval()
x394 = torch.randn(torch.Size([1, 56, 1, 1]))
x391 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x394, x391)
end = time.time()
print(end-start)
