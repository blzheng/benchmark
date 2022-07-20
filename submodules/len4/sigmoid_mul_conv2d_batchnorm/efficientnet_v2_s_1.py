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
        self.sigmoid1 = Sigmoid()
        self.conv2d28 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x89, x85):
        x90=self.sigmoid1(x89)
        x91=operator.mul(x90, x85)
        x92=self.conv2d28(x91)
        x93=self.batchnorm2d24(x92)
        return x93

m = M().eval()
x89 = torch.randn(torch.Size([1, 512, 1, 1]))
x85 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x89, x85)
end = time.time()
print(end-start)
