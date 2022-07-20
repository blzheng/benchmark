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
        self.batchnorm2d10 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU6(inplace=True)
        self.conv2d11 = Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x29):
        x30=self.batchnorm2d10(x29)
        x31=self.relu67(x30)
        x32=self.conv2d11(x31)
        return x32

m = M().eval()
x29 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
