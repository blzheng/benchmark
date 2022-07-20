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
        self.relu32 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)
        self.conv2d45 = Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)

    def forward(self, x136):
        x137=self.relu32(x136)
        x138=self.conv2d44(x137)
        x139=self.batchnorm2d28(x138)
        x140=self.relu33(x139)
        x141=self.conv2d45(x140)
        return x141

m = M().eval()
x136 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x136)
end = time.time()
print(end-start)
