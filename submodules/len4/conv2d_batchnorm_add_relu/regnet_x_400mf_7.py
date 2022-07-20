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
        self.conv2d18 = Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)

    def forward(self, x55, x49):
        x56=self.conv2d18(x55)
        x57=self.batchnorm2d18(x56)
        x58=operator.add(x49, x57)
        x59=self.relu15(x58)
        return x59

m = M().eval()
x55 = torch.randn(torch.Size([1, 160, 14, 14]))
x49 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x55, x49)
end = time.time()
print(end-start)
