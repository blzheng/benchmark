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
        self.relu94 = ReLU(inplace=True)
        self.conv2d94 = Conv2d(608, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d95 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu95 = ReLU(inplace=True)

    def forward(self, x336):
        x337=self.relu94(x336)
        x338=self.conv2d94(x337)
        x339=self.batchnorm2d95(x338)
        x340=self.relu95(x339)
        return x340

m = M().eval()
x336 = torch.randn(torch.Size([1, 608, 7, 7]))
start = time.time()
output = m(x336)
end = time.time()
print(end-start)
