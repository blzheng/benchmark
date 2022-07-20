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
        self.conv2d195 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid39 = Sigmoid()
        self.conv2d196 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d116 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x612, x609):
        x613=self.conv2d195(x612)
        x614=self.sigmoid39(x613)
        x615=operator.mul(x614, x609)
        x616=self.conv2d196(x615)
        x617=self.batchnorm2d116(x616)
        return x617

m = M().eval()
x612 = torch.randn(torch.Size([1, 96, 1, 1]))
x609 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x612, x609)
end = time.time()
print(end-start)
