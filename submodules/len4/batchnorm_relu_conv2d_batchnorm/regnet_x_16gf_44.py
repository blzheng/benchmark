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
        self.batchnorm2d69 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d70 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x225):
        x226=self.batchnorm2d69(x225)
        x227=self.relu65(x226)
        x228=self.conv2d70(x227)
        x229=self.batchnorm2d70(x228)
        return x229

m = M().eval()
x225 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x225)
end = time.time()
print(end-start)
