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
        self.conv2d23 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x69, x55):
        x70=operator.add(x69, x55)
        x71=self.conv2d23(x70)
        x72=self.batchnorm2d13(x71)
        return x72

m = M().eval()
x69 = torch.randn(torch.Size([1, 40, 56, 56]))
x55 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x69, x55)
end = time.time()
print(end-start)
