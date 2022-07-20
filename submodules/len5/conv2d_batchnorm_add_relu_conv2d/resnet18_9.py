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
        self.conv2d17 = Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d17 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x50, x55):
        x56=self.conv2d17(x50)
        x57=self.batchnorm2d17(x56)
        x58=operator.add(x55, x57)
        x59=self.relu13(x58)
        x60=self.conv2d18(x59)
        return x60

m = M().eval()
x50 = torch.randn(torch.Size([1, 256, 14, 14]))
x55 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x50, x55)
end = time.time()
print(end-start)
