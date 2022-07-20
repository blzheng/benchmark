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
        self.conv2d111 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d112 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d80 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x362, x359):
        x363=self.conv2d111(x362)
        x364=self.sigmoid15(x363)
        x365=operator.mul(x364, x359)
        x366=self.conv2d112(x365)
        x367=self.batchnorm2d80(x366)
        return x367

m = M().eval()
x362 = torch.randn(torch.Size([1, 56, 1, 1]))
x359 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x362, x359)
end = time.time()
print(end-start)
