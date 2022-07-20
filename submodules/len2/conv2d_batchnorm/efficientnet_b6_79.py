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
        self.conv2d133 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d79 = BatchNorm2d(1200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x416):
        x417=self.conv2d133(x416)
        x418=self.batchnorm2d79(x417)
        return x418

m = M().eval()
x416 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x416)
end = time.time()
print(end-start)
