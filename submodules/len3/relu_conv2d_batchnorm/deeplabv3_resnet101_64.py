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
        self.relu64 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x227):
        x228=self.relu64(x227)
        x229=self.conv2d69(x228)
        x230=self.batchnorm2d69(x229)
        return x230

m = M().eval()
x227 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x227)
end = time.time()
print(end-start)
