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
        self.conv2d74 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x226, x211):
        x227=operator.add(x226, x211)
        x228=self.conv2d74(x227)
        x229=self.batchnorm2d44(x228)
        return x229

m = M().eval()
x226 = torch.randn(torch.Size([1, 136, 14, 14]))
x211 = torch.randn(torch.Size([1, 136, 14, 14]))
start = time.time()
output = m(x226, x211)
end = time.time()
print(end-start)
