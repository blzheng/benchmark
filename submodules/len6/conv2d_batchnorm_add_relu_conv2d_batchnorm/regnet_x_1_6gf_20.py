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
        self.conv2d55 = Conv2d(912, 912, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(912, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu51 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(912, 912, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(912, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x177, x171):
        x178=self.conv2d55(x177)
        x179=self.batchnorm2d55(x178)
        x180=operator.add(x171, x179)
        x181=self.relu51(x180)
        x182=self.conv2d56(x181)
        x183=self.batchnorm2d56(x182)
        return x183

m = M().eval()
x177 = torch.randn(torch.Size([1, 912, 7, 7]))
x171 = torch.randn(torch.Size([1, 912, 7, 7]))
start = time.time()
output = m(x177, x171)
end = time.time()
print(end-start)
