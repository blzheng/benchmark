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
        self.conv2d74 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x219, x205):
        x220=operator.add(x219, x205)
        x221=self.conv2d74(x220)
        x222=self.batchnorm2d44(x221)
        return x222

m = M().eval()
x219 = torch.randn(torch.Size([1, 112, 14, 14]))
x205 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x219, x205)
end = time.time()
print(end-start)
