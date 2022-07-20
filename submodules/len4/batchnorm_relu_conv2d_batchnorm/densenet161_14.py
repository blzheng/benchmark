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
        self.batchnorm2d30 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x108):
        x109=self.batchnorm2d30(x108)
        x110=self.relu30(x109)
        x111=self.conv2d30(x110)
        x112=self.batchnorm2d31(x111)
        return x112

m = M().eval()
x108 = torch.randn(torch.Size([1, 576, 28, 28]))
start = time.time()
output = m(x108)
end = time.time()
print(end-start)
