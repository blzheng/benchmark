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
        self.conv2d13 = Conv2d(72, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x38):
        x39=self.conv2d13(x38)
        x40=self.batchnorm2d11(x39)
        x41=self.conv2d14(x40)
        return x41

m = M().eval()
x38 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x38)
end = time.time()
print(end-start)
