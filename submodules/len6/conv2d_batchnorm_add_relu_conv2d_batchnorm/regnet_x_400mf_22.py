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
        self.conv2d61 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu57 = ReLU(inplace=True)
        self.conv2d62 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d62 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x197, x191):
        x198=self.conv2d61(x197)
        x199=self.batchnorm2d61(x198)
        x200=operator.add(x191, x199)
        x201=self.relu57(x200)
        x202=self.conv2d62(x201)
        x203=self.batchnorm2d62(x202)
        return x203

m = M().eval()
x197 = torch.randn(torch.Size([1, 400, 7, 7]))
x191 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x197, x191)
end = time.time()
print(end-start)
