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
        self.conv2d93 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(176, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x300):
        x301=self.conv2d93(x300)
        x302=self.batchnorm2d65(x301)
        return x302

m = M().eval()
x300 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x300)
end = time.time()
print(end-start)