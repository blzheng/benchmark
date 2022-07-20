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
        self.conv2d43 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu39 = ReLU(inplace=True)

    def forward(self, x137, x131):
        x138=self.conv2d43(x137)
        x139=self.batchnorm2d43(x138)
        x140=operator.add(x131, x139)
        x141=self.relu39(x140)
        return x141

m = M().eval()
x137 = torch.randn(torch.Size([1, 672, 7, 7]))
x131 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x137, x131)
end = time.time()
print(end-start)
