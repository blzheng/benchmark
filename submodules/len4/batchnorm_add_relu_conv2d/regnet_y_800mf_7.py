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
        self.batchnorm2d18 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x86, x73):
        x87=self.batchnorm2d18(x86)
        x88=operator.add(x73, x87)
        x89=self.relu20(x88)
        x90=self.conv2d29(x89)
        return x90

m = M().eval()
x86 = torch.randn(torch.Size([1, 320, 14, 14]))
x73 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x86, x73)
end = time.time()
print(end-start)
