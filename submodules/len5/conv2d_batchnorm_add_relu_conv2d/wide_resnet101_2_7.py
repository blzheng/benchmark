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
        self.conv2d20 = Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d21 = Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x64, x58):
        x65=self.conv2d20(x64)
        x66=self.batchnorm2d20(x65)
        x67=operator.add(x66, x58)
        x68=self.relu16(x67)
        x69=self.conv2d21(x68)
        return x69

m = M().eval()
x64 = torch.randn(torch.Size([1, 256, 28, 28]))
x58 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x64, x58)
end = time.time()
print(end-start)
