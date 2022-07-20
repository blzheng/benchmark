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
        self.conv2d147 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()
        self.conv2d148 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d96 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x468, x465):
        x469=self.conv2d147(x468)
        x470=self.sigmoid25(x469)
        x471=operator.mul(x470, x465)
        x472=self.conv2d148(x471)
        x473=self.batchnorm2d96(x472)
        return x473

m = M().eval()
x468 = torch.randn(torch.Size([1, 64, 1, 1]))
x465 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x468, x465)
end = time.time()
print(end-start)
