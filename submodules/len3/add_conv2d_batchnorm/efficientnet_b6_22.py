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
        self.conv2d138 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d82 = BatchNorm2d(1200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x431, x416):
        x432=operator.add(x431, x416)
        x433=self.conv2d138(x432)
        x434=self.batchnorm2d82(x433)
        return x434

m = M().eval()
x431 = torch.randn(torch.Size([1, 200, 14, 14]))
x416 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x431, x416)
end = time.time()
print(end-start)
