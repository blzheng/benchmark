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
        self.conv2d42 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu39 = ReLU(inplace=True)

    def forward(self, x135, x129):
        x136=self.conv2d42(x135)
        x137=self.batchnorm2d42(x136)
        x138=operator.add(x129, x137)
        x139=self.relu39(x138)
        return x139

m = M().eval()
x135 = torch.randn(torch.Size([1, 1344, 14, 14]))
x129 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x135, x129)
end = time.time()
print(end-start)
