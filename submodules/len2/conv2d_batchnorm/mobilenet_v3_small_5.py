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
        self.conv2d7 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x20):
        x21=self.conv2d7(x20)
        x22=self.batchnorm2d5(x21)
        return x22

m = M().eval()
x20 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x20)
end = time.time()
print(end-start)
