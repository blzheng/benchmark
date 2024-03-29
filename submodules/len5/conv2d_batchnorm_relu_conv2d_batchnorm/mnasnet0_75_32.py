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
        self.conv2d48 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(864, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(864, 864, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=864, bias=False)
        self.batchnorm2d49 = BatchNorm2d(864, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x138):
        x139=self.conv2d48(x138)
        x140=self.batchnorm2d48(x139)
        x141=self.relu32(x140)
        x142=self.conv2d49(x141)
        x143=self.batchnorm2d49(x142)
        return x143

m = M().eval()
x138 = torch.randn(torch.Size([1, 144, 7, 7]))
start = time.time()
output = m(x138)
end = time.time()
print(end-start)
