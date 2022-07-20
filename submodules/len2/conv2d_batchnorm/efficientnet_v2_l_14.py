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
        self.conv2d14 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x52):
        x53=self.conv2d14(x52)
        x54=self.batchnorm2d14(x53)
        return x54

m = M().eval()
x52 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x52)
end = time.time()
print(end-start)
