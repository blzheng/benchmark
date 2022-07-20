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
        self.conv2d28 = Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d29 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x85):
        x86=self.conv2d28(x85)
        x87=self.batchnorm2d16(x86)
        x88=self.conv2d29(x87)
        return x88

m = M().eval()
x85 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x85)
end = time.time()
print(end-start)
