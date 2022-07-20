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
        self.relu18 = ReLU(inplace=True)
        self.conv2d22 = Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x68):
        x69=self.relu18(x68)
        x70=self.conv2d22(x69)
        x71=self.batchnorm2d22(x70)
        return x71

m = M().eval()
x68 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x68)
end = time.time()
print(end-start)
