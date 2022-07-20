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
        self.batchnorm2d7 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(240, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x26):
        x27=self.batchnorm2d7(x26)
        x28=self.relu7(x27)
        x29=self.conv2d7(x28)
        x30=self.batchnorm2d8(x29)
        return x30

m = M().eval()
x26 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x26)
end = time.time()
print(end-start)
