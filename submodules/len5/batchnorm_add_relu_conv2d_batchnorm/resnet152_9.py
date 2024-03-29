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
        self.batchnorm2d26 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)
        self.conv2d27 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x85, x78):
        x86=self.batchnorm2d26(x85)
        x87=operator.add(x86, x78)
        x88=self.relu22(x87)
        x89=self.conv2d27(x88)
        x90=self.batchnorm2d27(x89)
        return x90

m = M().eval()
x85 = torch.randn(torch.Size([1, 512, 28, 28]))
x78 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x85, x78)
end = time.time()
print(end-start)
