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
        self.relu12 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        self.batchnorm2d19 = BatchNorm2d(120, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(40, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x53):
        x54=self.relu12(x53)
        x55=self.conv2d19(x54)
        x56=self.batchnorm2d19(x55)
        x57=self.relu13(x56)
        x58=self.conv2d20(x57)
        x59=self.batchnorm2d20(x58)
        return x59

m = M().eval()
x53 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x53)
end = time.time()
print(end-start)
