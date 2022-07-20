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
        self.conv2d26 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)

    def forward(self, x84, x88):
        x85=self.conv2d26(x84)
        x86=self.batchnorm2d26(x85)
        x89=operator.add(x86, x88)
        x90=self.relu22(x89)
        return x90

m = M().eval()
x84 = torch.randn(torch.Size([1, 1024, 14, 14]))
x88 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x84, x88)
end = time.time()
print(end-start)
