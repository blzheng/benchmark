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
        self.conv2d32 = Conv2d(416, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)

    def forward(self, x117):
        x118=self.conv2d32(x117)
        x119=self.batchnorm2d33(x118)
        x120=self.relu33(x119)
        return x120

m = M().eval()
x117 = torch.randn(torch.Size([1, 416, 28, 28]))
start = time.time()
output = m(x117)
end = time.time()
print(end-start)
