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
        self.conv2d21 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x60):
        x61=self.conv2d21(x60)
        x62=self.batchnorm2d15(x61)
        return x62

m = M().eval()
x60 = torch.randn(torch.Size([1, 40, 14, 14]))
start = time.time()
output = m(x60)
end = time.time()
print(end-start)
