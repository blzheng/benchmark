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
        self.relu16 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)

    def forward(self, x60):
        x61=self.relu16(x60)
        x62=self.conv2d16(x61)
        x63=self.batchnorm2d17(x62)
        x64=self.relu17(x63)
        return x64

m = M().eval()
x60 = torch.randn(torch.Size([1, 160, 28, 28]))
start = time.time()
output = m(x60)
end = time.time()
print(end-start)
