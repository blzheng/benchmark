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
        self.conv2d20 = Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(32, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x57, x51):
        x58=self.conv2d20(x57)
        x59=self.batchnorm2d20(x58)
        x60=operator.add(x59, x51)
        return x60

m = M().eval()
x57 = torch.randn(torch.Size([1, 96, 28, 28]))
x51 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x57, x51)
end = time.time()
print(end-start)
