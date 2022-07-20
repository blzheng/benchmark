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
        self.conv2d18 = Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)

    def forward(self, x51):
        x52=self.conv2d18(x51)
        x53=self.batchnorm2d18(x52)
        x54=self.relu12(x53)
        return x54

m = M().eval()
x51 = torch.randn(torch.Size([1, 24, 28, 28]))
start = time.time()
output = m(x51)
end = time.time()
print(end-start)
