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
        self.conv2d10 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x38):
        x39=self.conv2d10(x38)
        x40=self.batchnorm2d10(x39)
        return x40

m = M().eval()
x38 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x38)
end = time.time()
print(end-start)
