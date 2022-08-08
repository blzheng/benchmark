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
        self.conv2d22 = Conv2d(168, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(408, 408, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=17, bias=False)
        self.batchnorm2d23 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d24 = Conv2d(408, 408, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x67):
        x70=self.conv2d22(x67)
        x71=self.batchnorm2d22(x70)
        x72=self.relu19(x71)
        x73=self.conv2d23(x72)
        x74=self.batchnorm2d23(x73)
        x75=self.relu20(x74)
        x76=self.conv2d24(x75)
        return x76

m = M().eval()
x67 = torch.randn(torch.Size([1, 168, 28, 28]))
start = time.time()
output = m(x67)
end = time.time()
print(end-start)
