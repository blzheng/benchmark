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
        self.conv2d18 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d20 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x60):
        x61=self.conv2d18(x60)
        x62=self.batchnorm2d18(x61)
        x63=self.relu16(x62)
        x64=self.conv2d19(x63)
        x65=self.batchnorm2d19(x64)
        x66=self.relu16(x65)
        x67=self.conv2d20(x66)
        x68=self.batchnorm2d20(x67)
        return x68

m = M().eval()
x60 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x60)
end = time.time()
print(end-start)
