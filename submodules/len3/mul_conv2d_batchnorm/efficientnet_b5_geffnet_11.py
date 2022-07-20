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
        self.conv2d57 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x165, x170):
        x171=operator.mul(x165, x170)
        x172=self.conv2d57(x171)
        x173=self.batchnorm2d33(x172)
        return x173

m = M().eval()
x165 = torch.randn(torch.Size([1, 384, 28, 28]))
x170 = torch.randn(torch.Size([1, 384, 1, 1]))
start = time.time()
output = m(x165, x170)
end = time.time()
print(end-start)