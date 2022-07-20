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
        self.conv2d21 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x78, x51, x58, x65, x72, x86):
        x79=self.conv2d21(x78)
        x87=torch.cat([x51, x58, x65, x72, x79, x86], 1)
        x88=self.batchnorm2d24(x87)
        return x88

m = M().eval()
x78 = torch.randn(torch.Size([1, 128, 28, 28]))
x51 = torch.randn(torch.Size([1, 128, 28, 28]))
x58 = torch.randn(torch.Size([1, 32, 28, 28]))
x65 = torch.randn(torch.Size([1, 32, 28, 28]))
x72 = torch.randn(torch.Size([1, 32, 28, 28]))
x86 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x78, x51, x58, x65, x72, x86)
end = time.time()
print(end-start)
