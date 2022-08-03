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
        self.conv2d15 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(120, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        self.batchnorm2d16 = BatchNorm2d(120, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)

    def forward(self, x42):
        x43=self.conv2d15(x42)
        x44=self.batchnorm2d15(x43)
        x45=self.relu10(x44)
        x46=self.conv2d16(x45)
        x47=self.batchnorm2d16(x46)
        x48=self.relu11(x47)
        return x48

m = M().eval()
x42 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x42)
end = time.time()
print(end-start)
