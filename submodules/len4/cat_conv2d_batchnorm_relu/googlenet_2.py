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
        self.conv2d21 = Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x68, x74, x80, x84):
        x85=torch.cat([x68, x74, x80, x84], 1)
        x86=self.conv2d21(x85)
        x87=self.batchnorm2d21(x86)
        x88=torch.nn.functional.relu(x87,inplace=True)
        return x88

m = M().eval()
x68 = torch.randn(torch.Size([1, 192, 14, 14]))
x74 = torch.randn(torch.Size([1, 208, 14, 14]))
x80 = torch.randn(torch.Size([1, 48, 14, 14]))
x84 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x68, x74, x80, x84)
end = time.time()
print(end-start)
