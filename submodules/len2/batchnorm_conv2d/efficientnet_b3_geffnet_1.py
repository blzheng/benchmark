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
        self.batchnorm2d16 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d29 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x85):
        x86=self.batchnorm2d16(x85)
        x87=self.conv2d29(x86)
        return x87

m = M().eval()
x85 = torch.randn(torch.Size([1, 48, 28, 28]))
start = time.time()
output = m(x85)
end = time.time()
print(end-start)
