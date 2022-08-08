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
        self.batchnorm2d45 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x149, x152):
        x150=self.batchnorm2d45(x149)
        x153=operator.add(x150, x152)
        x154=self.relu40(x153)
        x155=self.conv2d47(x154)
        return x155

m = M().eval()
x149 = torch.randn(torch.Size([1, 2048, 28, 28]))
x152 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x149, x152)
end = time.time()
print(end-start)
