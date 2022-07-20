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
        self.relu48 = ReLU(inplace=True)
        self.conv2d64 = Conv2d(320, 784, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d40 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x185, x199):
        x200=operator.add(x185, x199)
        x201=self.relu48(x200)
        x202=self.conv2d64(x201)
        x203=self.batchnorm2d40(x202)
        return x203

m = M().eval()
x185 = torch.randn(torch.Size([1, 320, 14, 14]))
x199 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x185, x199)
end = time.time()
print(end-start)
