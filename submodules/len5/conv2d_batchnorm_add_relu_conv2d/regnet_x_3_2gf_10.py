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
        self.conv2d27 = Conv2d(192, 432, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d27 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)
        self.conv2d31 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x87, x97):
        x88=self.conv2d27(x87)
        x89=self.batchnorm2d27(x88)
        x98=operator.add(x89, x97)
        x99=self.relu27(x98)
        x100=self.conv2d31(x99)
        return x100

m = M().eval()
x87 = torch.randn(torch.Size([1, 192, 28, 28]))
x97 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x87, x97)
end = time.time()
print(end-start)
