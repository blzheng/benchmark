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
        self.conv2d81 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d81 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu76 = ReLU(inplace=True)
        self.conv2d82 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x266, x260):
        x267=self.conv2d81(x266)
        x268=self.batchnorm2d81(x267)
        x269=operator.add(x268, x260)
        x270=self.relu76(x269)
        x271=self.conv2d82(x270)
        return x271

m = M().eval()
x266 = torch.randn(torch.Size([1, 512, 14, 14]))
x260 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x266, x260)
end = time.time()
print(end-start)
