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
        self.conv2d22 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x74, x71):
        x75=self.conv2d22(x74)
        x76=self.batchnorm2d22(x75)
        x77=operator.add(x76, x71)
        x78=self.relu19(x77)
        x79=self.conv2d23(x78)
        x80=self.batchnorm2d23(x79)
        return x80

m = M().eval()
x74 = torch.randn(torch.Size([1, 256, 14, 14]))
x71 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x74, x71)
end = time.time()
print(end-start)
