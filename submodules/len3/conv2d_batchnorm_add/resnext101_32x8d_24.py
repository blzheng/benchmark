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
        self.conv2d69 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x226, x220):
        x227=self.conv2d69(x226)
        x228=self.batchnorm2d69(x227)
        x229=operator.add(x228, x220)
        return x229

m = M().eval()
x226 = torch.randn(torch.Size([1, 1024, 14, 14]))
x220 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x226, x220)
end = time.time()
print(end-start)
