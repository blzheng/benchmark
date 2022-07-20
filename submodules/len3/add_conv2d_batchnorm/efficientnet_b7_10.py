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
        self.conv2d67 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x208, x193):
        x209=operator.add(x208, x193)
        x210=self.conv2d67(x209)
        x211=self.batchnorm2d39(x210)
        return x211

m = M().eval()
x208 = torch.randn(torch.Size([1, 80, 28, 28]))
x193 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x208, x193)
end = time.time()
print(end-start)
