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
        self.batchnorm2d4 = BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d5 = Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x17):
        x18=self.batchnorm2d4(x17)
        x19=self.conv2d5(x18)
        x20=self.batchnorm2d5(x19)
        return x20

m = M().eval()
x17 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x17)
end = time.time()
print(end-start)
