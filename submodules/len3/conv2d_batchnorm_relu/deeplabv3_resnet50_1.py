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
        self.conv2d1 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)

    def forward(self, x6):
        x7=self.conv2d1(x6)
        x8=self.batchnorm2d1(x7)
        x9=self.relu1(x8)
        return x9

m = M().eval()
x6 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x6)
end = time.time()
print(end-start)
