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
        self.batchnorm2d17 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x84, x71):
        x85=self.batchnorm2d17(x84)
        x86=operator.add(x71, x85)
        x87=self.relu20(x86)
        x88=self.conv2d28(x87)
        return x88

m = M().eval()
x84 = torch.randn(torch.Size([1, 120, 28, 28]))
x71 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x84, x71)
end = time.time()
print(end-start)
