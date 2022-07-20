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
        self.conv2d17 = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x62, x56):
        x63=operator.add(x62, x56)
        x64=self.conv2d17(x63)
        x65=self.batchnorm2d17(x64)
        return x65

m = M().eval()
x62 = torch.randn(torch.Size([1, 64, 56, 56]))
x56 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x62, x56)
end = time.time()
print(end-start)
