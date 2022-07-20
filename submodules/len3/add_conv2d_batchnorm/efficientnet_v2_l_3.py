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
        self.conv2d5 = Conv2d(32, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x22, x18):
        x23=operator.add(x22, x18)
        x24=self.conv2d5(x23)
        x25=self.batchnorm2d5(x24)
        return x25

m = M().eval()
x22 = torch.randn(torch.Size([1, 32, 112, 112]))
x18 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x22, x18)
end = time.time()
print(end-start)
