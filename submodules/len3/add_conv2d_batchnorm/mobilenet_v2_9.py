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
        self.conv2d48 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x129, x137):
        x138=operator.add(x129, x137)
        x139=self.conv2d48(x138)
        x140=self.batchnorm2d48(x139)
        return x140

m = M().eval()
x129 = torch.randn(torch.Size([1, 160, 7, 7]))
x137 = torch.randn(torch.Size([1, 160, 7, 7]))
start = time.time()
output = m(x129, x137)
end = time.time()
print(end-start)
