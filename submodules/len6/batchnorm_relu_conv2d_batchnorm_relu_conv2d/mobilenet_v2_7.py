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
        self.batchnorm2d21 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu614 = ReLU6(inplace=True)
        self.conv2d22 = Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.batchnorm2d22 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu615 = ReLU6(inplace=True)
        self.conv2d23 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x60):
        x61=self.batchnorm2d21(x60)
        x62=self.relu614(x61)
        x63=self.conv2d22(x62)
        x64=self.batchnorm2d22(x63)
        x65=self.relu615(x64)
        x66=self.conv2d23(x65)
        return x66

m = M().eval()
x60 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x60)
end = time.time()
print(end-start)
