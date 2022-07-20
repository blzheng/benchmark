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
        self.batchnorm2d3 = BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d4 = Conv2d(80, 192, kernel_size=(3, 3), stride=(1, 1), bias=False)

    def forward(self, x24):
        x25=self.batchnorm2d3(x24)
        x26=torch.nn.functional.relu(x25,inplace=True)
        x27=self.conv2d4(x26)
        return x27

m = M().eval()
x24 = torch.randn(torch.Size([1, 80, 54, 54]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)