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
        self.conv2d17 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x56, x50):
        x57=self.conv2d17(x56)
        x58=self.batchnorm2d17(x57)
        x59=operator.add(x58, x50)
        x60=self.relu13(x59)
        x61=self.conv2d18(x60)
        x62=self.batchnorm2d18(x61)
        return x62

m = M().eval()
x56 = torch.randn(torch.Size([1, 128, 28, 28]))
x50 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x56, x50)
end = time.time()
print(end-start)
