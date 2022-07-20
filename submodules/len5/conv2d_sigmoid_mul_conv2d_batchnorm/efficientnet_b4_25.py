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
        self.conv2d127 = Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()
        self.conv2d128 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x394, x391):
        x395=self.conv2d127(x394)
        x396=self.sigmoid25(x395)
        x397=operator.mul(x396, x391)
        x398=self.conv2d128(x397)
        x399=self.batchnorm2d76(x398)
        return x399

m = M().eval()
x394 = torch.randn(torch.Size([1, 68, 1, 1]))
x391 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x394, x391)
end = time.time()
print(end-start)
