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
        self.conv2d9 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x26, x14):
        x27=operator.add(x26, x14)
        x28=self.conv2d9(x27)
        x29=self.batchnorm2d5(x28)
        return x29

m = M().eval()
x26 = torch.randn(torch.Size([1, 24, 112, 112]))
x14 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x26, x14)
end = time.time()
print(end-start)
