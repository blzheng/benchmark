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
        self.sigmoid23 = Sigmoid()
        self.conv2d116 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x361, x357):
        x362=self.sigmoid23(x361)
        x363=operator.mul(x362, x357)
        x364=self.conv2d116(x363)
        x365=self.batchnorm2d68(x364)
        return x365

m = M().eval()
x361 = torch.randn(torch.Size([1, 960, 1, 1]))
x357 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x361, x357)
end = time.time()
print(end-start)