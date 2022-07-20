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
        self.maxpool2d4 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2d15 = Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x64):
        x65=self.maxpool2d4(x64)
        x66=self.conv2d15(x65)
        x67=self.batchnorm2d15(x66)
        return x67

m = M().eval()
x64 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x64)
end = time.time()
print(end-start)
