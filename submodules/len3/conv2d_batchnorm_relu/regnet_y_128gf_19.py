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
        self.conv2d49 = Conv2d(1056, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)

    def forward(self, x151):
        x154=self.conv2d49(x151)
        x155=self.batchnorm2d31(x154)
        x156=self.relu37(x155)
        return x156

m = M().eval()
x151 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x151)
end = time.time()
print(end-start)
