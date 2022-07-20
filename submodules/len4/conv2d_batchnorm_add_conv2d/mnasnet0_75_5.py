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
        self.conv2d47 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(144, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d48 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x135, x129):
        x136=self.conv2d47(x135)
        x137=self.batchnorm2d47(x136)
        x138=operator.add(x137, x129)
        x139=self.conv2d48(x138)
        return x139

m = M().eval()
x135 = torch.randn(torch.Size([1, 864, 7, 7]))
x129 = torch.randn(torch.Size([1, 144, 7, 7]))
start = time.time()
output = m(x135, x129)
end = time.time()
print(end-start)
