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
        self.conv2d111 = Conv2d(1408, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d112 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x395):
        x396=self.conv2d111(x395)
        x397=self.batchnorm2d112(x396)
        return x397

m = M().eval()
x395 = torch.randn(torch.Size([1, 1408, 14, 14]))
start = time.time()
output = m(x395)
end = time.time()
print(end-start)
