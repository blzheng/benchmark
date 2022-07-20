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
        self.conv2d30 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)

    def forward(self, x102, x106):
        x103=self.conv2d30(x102)
        x104=self.batchnorm2d30(x103)
        x107=operator.add(x104, x106)
        x108=self.relu27(x107)
        return x108

m = M().eval()
x102 = torch.randn(torch.Size([1, 512, 7, 7]))
x106 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x102, x106)
end = time.time()
print(end-start)
