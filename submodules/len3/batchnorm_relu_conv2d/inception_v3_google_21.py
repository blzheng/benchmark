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
        self.batchnorm2d42 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d43 = Conv2d(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)

    def forward(self, x152):
        x153=self.batchnorm2d42(x152)
        x154=torch.nn.functional.relu(x153,inplace=True)
        x155=self.conv2d43(x154)
        return x155

m = M().eval()
x152 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
