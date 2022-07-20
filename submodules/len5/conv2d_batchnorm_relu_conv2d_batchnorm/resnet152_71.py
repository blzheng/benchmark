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
        self.conv2d110 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d110 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu106 = ReLU(inplace=True)
        self.conv2d111 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d111 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x363):
        x364=self.conv2d110(x363)
        x365=self.batchnorm2d110(x364)
        x366=self.relu106(x365)
        x367=self.conv2d111(x366)
        x368=self.batchnorm2d111(x367)
        return x368

m = M().eval()
x363 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x363)
end = time.time()
print(end-start)
