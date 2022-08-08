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
        self.conv2d11 = Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(232, 696, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x32, x27, x21):
        x33=operator.mul(x32, x27)
        x34=self.conv2d11(x33)
        x35=self.batchnorm2d7(x34)
        x36=operator.add(x21, x35)
        x37=self.relu8(x36)
        x38=self.conv2d12(x37)
        return x38

m = M().eval()
x32 = torch.randn(torch.Size([1, 232, 1, 1]))
x27 = torch.randn(torch.Size([1, 232, 56, 56]))
x21 = torch.randn(torch.Size([1, 232, 56, 56]))
start = time.time()
output = m(x32, x27, x21)
end = time.time()
print(end-start)
