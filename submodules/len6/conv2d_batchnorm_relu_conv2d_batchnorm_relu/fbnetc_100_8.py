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
        self.conv2d22 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d23 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)

    def forward(self, x71):
        x72=self.conv2d22(x71)
        x73=self.batchnorm2d22(x72)
        x74=self.relu15(x73)
        x75=self.conv2d23(x74)
        x76=self.batchnorm2d23(x75)
        x77=self.relu16(x76)
        return x77

m = M().eval()
x71 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x71)
end = time.time()
print(end-start)
