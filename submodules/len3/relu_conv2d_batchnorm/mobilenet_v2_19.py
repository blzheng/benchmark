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
        self.relu619 = ReLU6(inplace=True)
        self.conv2d29 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x82):
        x83=self.relu619(x82)
        x84=self.conv2d29(x83)
        x85=self.batchnorm2d29(x84)
        return x85

m = M().eval()
x82 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x82)
end = time.time()
print(end-start)
