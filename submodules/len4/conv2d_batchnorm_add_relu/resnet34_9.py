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
        self.conv2d18 = Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d18 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)

    def forward(self, x55, x60):
        x61=self.conv2d18(x55)
        x62=self.batchnorm2d18(x61)
        x63=operator.add(x60, x62)
        x64=self.relu15(x63)
        return x64

m = M().eval()
x55 = torch.randn(torch.Size([1, 128, 28, 28]))
x60 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x55, x60)
end = time.time()
print(end-start)
