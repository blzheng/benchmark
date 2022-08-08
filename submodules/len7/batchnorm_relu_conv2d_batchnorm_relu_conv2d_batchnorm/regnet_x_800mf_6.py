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
        self.batchnorm2d22 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=18, bias=False)
        self.batchnorm2d23 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x70):
        x71=self.batchnorm2d22(x70)
        x72=self.relu19(x71)
        x73=self.conv2d23(x72)
        x74=self.batchnorm2d23(x73)
        x75=self.relu20(x74)
        x76=self.conv2d24(x75)
        x77=self.batchnorm2d24(x76)
        return x77

m = M().eval()
x70 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x70)
end = time.time()
print(end-start)
