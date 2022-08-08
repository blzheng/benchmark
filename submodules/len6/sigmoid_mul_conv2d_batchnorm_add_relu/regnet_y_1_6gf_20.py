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
        self.sigmoid20 = Sigmoid()
        self.conv2d108 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu84 = ReLU(inplace=True)

    def forward(self, x339, x335, x329):
        x340=self.sigmoid20(x339)
        x341=operator.mul(x340, x335)
        x342=self.conv2d108(x341)
        x343=self.batchnorm2d66(x342)
        x344=operator.add(x329, x343)
        x345=self.relu84(x344)
        return x345

m = M().eval()
x339 = torch.randn(torch.Size([1, 336, 1, 1]))
x335 = torch.randn(torch.Size([1, 336, 14, 14]))
x329 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x339, x335, x329)
end = time.time()
print(end-start)
