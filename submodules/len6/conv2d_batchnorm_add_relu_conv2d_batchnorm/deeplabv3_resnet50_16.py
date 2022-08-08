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
        self.conv2d45 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x148, x152):
        x149=self.conv2d45(x148)
        x150=self.batchnorm2d45(x149)
        x153=operator.add(x150, x152)
        x154=self.relu40(x153)
        x155=self.conv2d47(x154)
        x156=self.batchnorm2d47(x155)
        return x156

m = M().eval()
x148 = torch.randn(torch.Size([1, 512, 28, 28]))
x152 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x148, x152)
end = time.time()
print(end-start)
