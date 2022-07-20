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
        self.conv2d23 = Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x88):
        x89=self.conv2d23(x88)
        x90=self.batchnorm2d23(x89)
        return x90

m = M().eval()
x88 = torch.randn(torch.Size([1, 64, 25, 25]))
start = time.time()
output = m(x88)
end = time.time()
print(end-start)
