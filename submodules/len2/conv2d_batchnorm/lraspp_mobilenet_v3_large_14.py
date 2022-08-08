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
        self.conv2d18 = Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x52):
        x53=self.conv2d18(x52)
        x54=self.batchnorm2d14(x53)
        return x54

m = M().eval()
x52 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x52)
end = time.time()
print(end-start)
