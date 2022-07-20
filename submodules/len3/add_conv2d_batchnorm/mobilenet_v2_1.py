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
        self.conv2d15 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x33, x41):
        x42=operator.add(x33, x41)
        x43=self.conv2d15(x42)
        x44=self.batchnorm2d15(x43)
        return x44

m = M().eval()
x33 = torch.randn(torch.Size([1, 32, 28, 28]))
x41 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x33, x41)
end = time.time()
print(end-start)
