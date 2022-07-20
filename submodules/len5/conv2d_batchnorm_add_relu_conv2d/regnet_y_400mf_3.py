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
        self.conv2d12 = Conv2d(104, 104, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu8 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(104, 104, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x35, x23):
        x36=self.conv2d12(x35)
        x37=self.batchnorm2d8(x36)
        x38=operator.add(x23, x37)
        x39=self.relu8(x38)
        x40=self.conv2d13(x39)
        return x40

m = M().eval()
x35 = torch.randn(torch.Size([1, 104, 28, 28]))
x23 = torch.randn(torch.Size([1, 104, 28, 28]))
start = time.time()
output = m(x35, x23)
end = time.time()
print(end-start)
