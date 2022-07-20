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
        self.conv2d73 = Conv2d(720, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x210, x215):
        x216=operator.mul(x210, x215)
        x217=self.conv2d73(x216)
        x218=self.batchnorm2d43(x217)
        return x218

m = M().eval()
x210 = torch.randn(torch.Size([1, 720, 14, 14]))
x215 = torch.randn(torch.Size([1, 720, 1, 1]))
start = time.time()
output = m(x210, x215)
end = time.time()
print(end-start)
