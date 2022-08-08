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
        self.conv2d25 = Conv2d(120, 30, kernel_size=(1, 1), stride=(1, 1))
        self.relu19 = ReLU()
        self.conv2d26 = Conv2d(30, 120, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()
        self.conv2d27 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x78, x77, x71):
        x79=self.conv2d25(x78)
        x80=self.relu19(x79)
        x81=self.conv2d26(x80)
        x82=self.sigmoid4(x81)
        x83=operator.mul(x82, x77)
        x84=self.conv2d27(x83)
        x85=self.batchnorm2d17(x84)
        x86=operator.add(x71, x85)
        return x86

m = M().eval()
x78 = torch.randn(torch.Size([1, 120, 1, 1]))
x77 = torch.randn(torch.Size([1, 120, 28, 28]))
x71 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x78, x77, x71)
end = time.time()
print(end-start)
