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
        self.conv2d153 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d153 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu148 = ReLU(inplace=True)
        self.conv2d154 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d154 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x505):
        x506=self.conv2d153(x505)
        x507=self.batchnorm2d153(x506)
        x508=self.relu148(x507)
        x509=self.conv2d154(x508)
        x510=self.batchnorm2d154(x509)
        return x510

m = M().eval()
x505 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x505)
end = time.time()
print(end-start)
