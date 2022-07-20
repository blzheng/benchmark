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
        self.conv2d178 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d106 = BatchNorm2d(1824, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x558):
        x559=self.conv2d178(x558)
        x560=self.batchnorm2d106(x559)
        return x560

m = M().eval()
x558 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x558)
end = time.time()
print(end-start)
