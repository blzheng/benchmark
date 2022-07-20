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
        self.relu622 = ReLU6(inplace=True)
        self.conv2d34 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d34 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu623 = ReLU6(inplace=True)
        self.conv2d35 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x96):
        x97=self.relu622(x96)
        x98=self.conv2d34(x97)
        x99=self.batchnorm2d34(x98)
        x100=self.relu623(x99)
        x101=self.conv2d35(x100)
        return x101

m = M().eval()
x96 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x96)
end = time.time()
print(end-start)
