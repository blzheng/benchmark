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
        self.conv2d79 = Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x240):
        x241=self.conv2d79(x240)
        x242=self.batchnorm2d47(x241)
        return x242

m = M().eval()
x240 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x240)
end = time.time()
print(end-start)
