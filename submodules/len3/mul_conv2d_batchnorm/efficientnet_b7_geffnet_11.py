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
        self.conv2d56 = Conv2d(288, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x164, x169):
        x170=operator.mul(x164, x169)
        x171=self.conv2d56(x170)
        x172=self.batchnorm2d32(x171)
        return x172

m = M().eval()
x164 = torch.randn(torch.Size([1, 288, 28, 28]))
x169 = torch.randn(torch.Size([1, 288, 1, 1]))
start = time.time()
output = m(x164, x169)
end = time.time()
print(end-start)
