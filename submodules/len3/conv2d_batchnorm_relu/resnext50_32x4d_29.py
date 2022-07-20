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
        self.conv2d47 = Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)

    def forward(self, x152):
        x153=self.conv2d47(x152)
        x154=self.batchnorm2d47(x153)
        x155=self.relu43(x154)
        return x155

m = M().eval()
x152 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
