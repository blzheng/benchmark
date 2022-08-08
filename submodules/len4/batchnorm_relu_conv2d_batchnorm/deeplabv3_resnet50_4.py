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
        self.batchnorm2d8 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x29):
        x30=self.batchnorm2d8(x29)
        x31=self.relu7(x30)
        x32=self.conv2d9(x31)
        x33=self.batchnorm2d9(x32)
        return x33

m = M().eval()
x29 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
