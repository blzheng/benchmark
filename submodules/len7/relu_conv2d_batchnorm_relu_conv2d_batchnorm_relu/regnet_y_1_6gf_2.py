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
        self.relu16 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(120, 120, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=5, bias=False)
        self.batchnorm2d16 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)

    def forward(self, x70):
        x71=self.relu16(x70)
        x72=self.conv2d23(x71)
        x73=self.batchnorm2d15(x72)
        x74=self.relu17(x73)
        x75=self.conv2d24(x74)
        x76=self.batchnorm2d16(x75)
        x77=self.relu18(x76)
        return x77

m = M().eval()
x70 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x70)
end = time.time()
print(end-start)
