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
        self.batchnorm2d86 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu82 = ReLU(inplace=True)
        self.conv2d87 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d87 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x286):
        x287=self.batchnorm2d86(x286)
        x288=self.relu82(x287)
        x289=self.conv2d87(x288)
        x290=self.batchnorm2d87(x289)
        return x290

m = M().eval()
x286 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x286)
end = time.time()
print(end-start)
