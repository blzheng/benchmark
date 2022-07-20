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
        self.batchnorm2d6 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
        self.batchnorm2d7 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)

    def forward(self, x18):
        x19=self.batchnorm2d6(x18)
        x20=self.relu3(x19)
        x21=self.conv2d7(x20)
        x22=self.batchnorm2d7(x21)
        x23=self.relu4(x22)
        return x23

m = M().eval()
x18 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
