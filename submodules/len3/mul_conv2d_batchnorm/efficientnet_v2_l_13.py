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
        self.conv2d102 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d74 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x332, x327):
        x333=operator.mul(x332, x327)
        x334=self.conv2d102(x333)
        x335=self.batchnorm2d74(x334)
        return x335

m = M().eval()
x332 = torch.randn(torch.Size([1, 1344, 1, 1]))
x327 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x332, x327)
end = time.time()
print(end-start)
