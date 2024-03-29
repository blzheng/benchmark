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
        self.batchnorm2d6 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d7 = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x27):
        x28=self.batchnorm2d6(x27)
        x29=self.conv2d7(x28)
        x30=self.batchnorm2d7(x29)
        return x30

m = M().eval()
x27 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x27)
end = time.time()
print(end-start)
