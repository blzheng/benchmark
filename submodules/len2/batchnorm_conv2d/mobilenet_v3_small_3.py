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
        self.batchnorm2d20 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d31 = Conv2d(48, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x88):
        x89=self.batchnorm2d20(x88)
        x90=self.conv2d31(x89)
        return x90

m = M().eval()
x88 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x88)
end = time.time()
print(end-start)
