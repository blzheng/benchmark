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
        self.batchnorm2d49 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d84 = Conv2d(208, 1248, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x247):
        x248=self.batchnorm2d49(x247)
        x249=self.conv2d84(x248)
        x250=self.batchnorm2d50(x249)
        return x250

m = M().eval()
x247 = torch.randn(torch.Size([1, 208, 7, 7]))
start = time.time()
output = m(x247)
end = time.time()
print(end-start)
