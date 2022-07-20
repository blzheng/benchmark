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
        self.relu6 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(72, 168, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d8 = BatchNorm2d(168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x15, x23):
        x24=operator.add(x15, x23)
        x25=self.relu6(x24)
        x26=self.conv2d8(x25)
        x27=self.batchnorm2d8(x26)
        return x27

m = M().eval()
x15 = torch.randn(torch.Size([1, 72, 56, 56]))
x23 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x15, x23)
end = time.time()
print(end-start)
