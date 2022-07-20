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
        self.relu142 = ReLU(inplace=True)
        self.conv2d142 = Conv2d(992, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d143 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu143 = ReLU(inplace=True)

    def forward(self, x504):
        x505=self.relu142(x504)
        x506=self.conv2d142(x505)
        x507=self.batchnorm2d143(x506)
        x508=self.relu143(x507)
        return x508

m = M().eval()
x504 = torch.randn(torch.Size([1, 992, 7, 7]))
start = time.time()
output = m(x504)
end = time.time()
print(end-start)
