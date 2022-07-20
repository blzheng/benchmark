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
        self.conv2d21 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x67):
        x68=self.conv2d21(x67)
        x69=self.batchnorm2d21(x68)
        return x69

m = M().eval()
x67 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x67)
end = time.time()
print(end-start)
