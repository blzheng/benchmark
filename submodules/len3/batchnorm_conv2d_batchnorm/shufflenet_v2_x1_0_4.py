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
        self.batchnorm2d13 = BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x79):
        x80=self.batchnorm2d13(x79)
        x81=self.conv2d14(x80)
        x82=self.batchnorm2d14(x81)
        return x82

m = M().eval()
x79 = torch.randn(torch.Size([1, 58, 28, 28]))
start = time.time()
output = m(x79)
end = time.time()
print(end-start)
