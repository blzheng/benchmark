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
        self.conv2d107 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(176, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x332):
        x333=self.conv2d107(x332)
        x334=self.batchnorm2d63(x333)
        return x334

m = M().eval()
x332 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x332)
end = time.time()
print(end-start)
