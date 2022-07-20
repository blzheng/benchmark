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
        self.batchnorm2d32 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d57 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x171):
        x172=self.batchnorm2d32(x171)
        x173=self.conv2d57(x172)
        x174=self.batchnorm2d33(x173)
        return x174

m = M().eval()
x171 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x171)
end = time.time()
print(end-start)
