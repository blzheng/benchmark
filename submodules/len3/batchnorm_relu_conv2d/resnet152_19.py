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
        self.batchnorm2d31 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x102):
        x103=self.batchnorm2d31(x102)
        x104=self.relu28(x103)
        x105=self.conv2d32(x104)
        return x105

m = M().eval()
x102 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x102)
end = time.time()
print(end-start)
