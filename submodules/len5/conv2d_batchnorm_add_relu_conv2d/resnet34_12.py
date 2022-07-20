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
        self.conv2d24 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d25 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x81, x78):
        x82=self.conv2d24(x81)
        x83=self.batchnorm2d24(x82)
        x84=operator.add(x83, x78)
        x85=self.relu21(x84)
        x86=self.conv2d25(x85)
        return x86

m = M().eval()
x81 = torch.randn(torch.Size([1, 256, 14, 14]))
x78 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x81, x78)
end = time.time()
print(end-start)
