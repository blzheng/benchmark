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
        self.conv2d63 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)

    def forward(self, x208, x202):
        x209=self.conv2d63(x208)
        x210=self.batchnorm2d63(x209)
        x211=operator.add(x210, x202)
        x212=self.relu58(x211)
        return x212

m = M().eval()
x208 = torch.randn(torch.Size([1, 256, 28, 28]))
x202 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x208, x202)
end = time.time()
print(end-start)
