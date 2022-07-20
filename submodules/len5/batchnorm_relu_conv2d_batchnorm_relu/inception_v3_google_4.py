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
        self.batchnorm2d8 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d9 = Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x40):
        x41=self.batchnorm2d8(x40)
        x42=torch.nn.functional.relu(x41,inplace=True)
        x43=self.conv2d9(x42)
        x44=self.batchnorm2d9(x43)
        x45=torch.nn.functional.relu(x44,inplace=True)
        return x45

m = M().eval()
x40 = torch.randn(torch.Size([1, 64, 25, 25]))
start = time.time()
output = m(x40)
end = time.time()
print(end-start)
