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
        self.batchnorm2d30 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x99):
        x100=self.batchnorm2d30(x99)
        x101=self.relu28(x100)
        x102=self.conv2d31(x101)
        return x102

m = M().eval()
x99 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x99)
end = time.time()
print(end-start)
