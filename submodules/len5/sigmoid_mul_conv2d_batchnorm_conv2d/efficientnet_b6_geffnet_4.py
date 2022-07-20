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
        self.conv2d157 = Conv2d(1200, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d93 = BatchNorm2d(344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d158 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x467, x463):
        x468=x467.sigmoid()
        x469=operator.mul(x463, x468)
        x470=self.conv2d157(x469)
        x471=self.batchnorm2d93(x470)
        x472=self.conv2d158(x471)
        return x472

m = M().eval()
x467 = torch.randn(torch.Size([1, 1200, 1, 1]))
x463 = torch.randn(torch.Size([1, 1200, 7, 7]))
start = time.time()
output = m(x467, x463)
end = time.time()
print(end-start)
