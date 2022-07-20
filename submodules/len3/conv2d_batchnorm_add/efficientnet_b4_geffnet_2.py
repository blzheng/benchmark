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
        self.conv2d78 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x232, x220):
        x233=self.conv2d78(x232)
        x234=self.batchnorm2d46(x233)
        x235=operator.add(x234, x220)
        return x235

m = M().eval()
x232 = torch.randn(torch.Size([1, 672, 14, 14]))
x220 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x232, x220)
end = time.time()
print(end-start)
