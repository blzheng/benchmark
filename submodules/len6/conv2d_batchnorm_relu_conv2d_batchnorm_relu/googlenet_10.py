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
        self.conv2d30 = Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d31 = Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x105):
        x115=self.conv2d30(x105)
        x116=self.batchnorm2d30(x115)
        x117=torch.nn.functional.relu(x116,inplace=True)
        x118=self.conv2d31(x117)
        x119=self.batchnorm2d31(x118)
        x120=torch.nn.functional.relu(x119,inplace=True)
        return x120

m = M().eval()
x105 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x105)
end = time.time()
print(end-start)
