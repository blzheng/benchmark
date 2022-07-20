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
        self.batchnorm2d55 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu55 = ReLU(inplace=True)
        self.conv2d55 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU(inplace=True)

    def forward(self, x197):
        x198=self.batchnorm2d55(x197)
        x199=self.relu55(x198)
        x200=self.conv2d55(x199)
        x201=self.batchnorm2d56(x200)
        x202=self.relu56(x201)
        return x202

m = M().eval()
x197 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x197)
end = time.time()
print(end-start)
