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
        self.batchnorm2d15 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d16 = Conv2d(176, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x95):
        x96=self.batchnorm2d15(x95)
        x97=self.conv2d16(x96)
        x98=self.batchnorm2d16(x97)
        return x98

m = M().eval()
x95 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x95)
end = time.time()
print(end-start)