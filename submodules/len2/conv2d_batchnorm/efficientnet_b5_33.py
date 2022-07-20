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
        self.conv2d57 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x176):
        x177=self.conv2d57(x176)
        x178=self.batchnorm2d33(x177)
        return x178

m = M().eval()
x176 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x176)
end = time.time()
print(end-start)
