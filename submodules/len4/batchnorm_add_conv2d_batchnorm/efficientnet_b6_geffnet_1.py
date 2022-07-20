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
        self.batchnorm2d42 = BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d73 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x217, x204):
        x218=self.batchnorm2d42(x217)
        x219=operator.add(x218, x204)
        x220=self.conv2d73(x219)
        x221=self.batchnorm2d43(x220)
        return x221

m = M().eval()
x217 = torch.randn(torch.Size([1, 72, 28, 28]))
x204 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x217, x204)
end = time.time()
print(end-start)
