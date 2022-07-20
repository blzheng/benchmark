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
        self.conv2d267 = Conv2d(640, 3840, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d159 = BatchNorm2d(3840, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x840, x825):
        x841=operator.add(x840, x825)
        x842=self.conv2d267(x841)
        x843=self.batchnorm2d159(x842)
        return x843

m = M().eval()
x840 = torch.randn(torch.Size([1, 640, 7, 7]))
x825 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x840, x825)
end = time.time()
print(end-start)
