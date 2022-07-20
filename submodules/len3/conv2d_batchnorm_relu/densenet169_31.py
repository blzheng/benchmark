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
        self.conv2d63 = Conv2d(640, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d64 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)

    def forward(self, x227):
        x228=self.conv2d63(x227)
        x229=self.batchnorm2d64(x228)
        x230=self.relu64(x229)
        return x230

m = M().eval()
x227 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x227)
end = time.time()
print(end-start)
