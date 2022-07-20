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
        self.conv2d17 = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x57, x51):
        x58=operator.add(x57, x51)
        x59=self.conv2d17(x58)
        x60=self.batchnorm2d17(x59)
        return x60

m = M().eval()
x57 = torch.randn(torch.Size([1, 64, 28, 28]))
x51 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x57, x51)
end = time.time()
print(end-start)
