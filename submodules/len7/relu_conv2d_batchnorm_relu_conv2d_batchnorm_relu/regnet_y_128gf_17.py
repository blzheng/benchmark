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
        self.relu80 = ReLU(inplace=True)
        self.conv2d104 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu81 = ReLU(inplace=True)
        self.conv2d105 = Conv2d(2904, 2904, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)
        self.batchnorm2d65 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu82 = ReLU(inplace=True)

    def forward(self, x328):
        x329=self.relu80(x328)
        x330=self.conv2d104(x329)
        x331=self.batchnorm2d64(x330)
        x332=self.relu81(x331)
        x333=self.conv2d105(x332)
        x334=self.batchnorm2d65(x333)
        x335=self.relu82(x334)
        return x335

m = M().eval()
x328 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x328)
end = time.time()
print(end-start)
