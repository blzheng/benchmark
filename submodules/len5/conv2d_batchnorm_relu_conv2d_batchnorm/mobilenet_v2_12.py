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
        self.conv2d18 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu612 = ReLU6(inplace=True)
        self.conv2d19 = Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d19 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x51):
        x52=self.conv2d18(x51)
        x53=self.batchnorm2d18(x52)
        x54=self.relu612(x53)
        x55=self.conv2d19(x54)
        x56=self.batchnorm2d19(x55)
        return x56

m = M().eval()
x51 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x51)
end = time.time()
print(end-start)
