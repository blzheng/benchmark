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
        self.conv2d101 = Conv2d(1872, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d102 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x360):
        x361=self.conv2d101(x360)
        x362=self.batchnorm2d102(x361)
        return x362

m = M().eval()
x360 = torch.randn(torch.Size([1, 1872, 14, 14]))
start = time.time()
output = m(x360)
end = time.time()
print(end-start)
