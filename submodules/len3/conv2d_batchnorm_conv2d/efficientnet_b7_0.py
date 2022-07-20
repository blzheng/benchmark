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
        self.conv2d4 = Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d5 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)

    def forward(self, x12):
        x13=self.conv2d4(x12)
        x14=self.batchnorm2d2(x13)
        x15=self.conv2d5(x14)
        return x15

m = M().eval()
x12 = torch.randn(torch.Size([1, 64, 112, 112]))
start = time.time()
output = m(x12)
end = time.time()
print(end-start)
