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
        self.conv2d59 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x180, x165):
        x181=operator.add(x180, x165)
        x182=self.conv2d59(x181)
        x183=self.batchnorm2d35(x182)
        return x183

m = M().eval()
x180 = torch.randn(torch.Size([1, 96, 14, 14]))
x165 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x180, x165)
end = time.time()
print(end-start)
