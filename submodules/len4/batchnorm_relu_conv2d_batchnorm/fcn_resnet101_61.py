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
        self.batchnorm2d95 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu91 = ReLU(inplace=True)
        self.conv2d96 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d96 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x316):
        x317=self.batchnorm2d95(x316)
        x318=self.relu91(x317)
        x319=self.conv2d96(x318)
        x320=self.batchnorm2d96(x319)
        return x320

m = M().eval()
x316 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x316)
end = time.time()
print(end-start)
