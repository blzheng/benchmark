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
        self.conv2d33 = Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x98):
        x99=self.conv2d33(x98)
        x100=self.batchnorm2d27(x99)
        return x100

m = M().eval()
x98 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x98)
end = time.time()
print(end-start)
