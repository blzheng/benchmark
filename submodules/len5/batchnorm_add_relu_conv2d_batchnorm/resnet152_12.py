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
        self.batchnorm2d35 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)
        self.conv2d36 = Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x115, x108):
        x116=self.batchnorm2d35(x115)
        x117=operator.add(x116, x108)
        x118=self.relu31(x117)
        x119=self.conv2d36(x118)
        x120=self.batchnorm2d36(x119)
        return x120

m = M().eval()
x115 = torch.randn(torch.Size([1, 512, 28, 28]))
x108 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x115, x108)
end = time.time()
print(end-start)
