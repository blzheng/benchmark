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
        self.conv2d19 = Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x57):
        x58=self.conv2d19(x57)
        x59=self.batchnorm2d15(x58)
        return x59

m = M().eval()
x57 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x57)
end = time.time()
print(end-start)
