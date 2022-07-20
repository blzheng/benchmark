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
        self.conv2d9 = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x34, x28):
        x35=operator.add(x34, x28)
        x36=self.conv2d9(x35)
        x37=self.batchnorm2d9(x36)
        return x37

m = M().eval()
x34 = torch.randn(torch.Size([1, 64, 56, 56]))
x28 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x34, x28)
end = time.time()
print(end-start)
