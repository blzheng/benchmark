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
        self.conv2d32 = Conv2d(448, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x98, x93):
        x99=operator.mul(x98, x93)
        x100=self.conv2d32(x99)
        x101=self.batchnorm2d20(x100)
        return x101

m = M().eval()
x98 = torch.randn(torch.Size([1, 448, 1, 1]))
x93 = torch.randn(torch.Size([1, 448, 28, 28]))
start = time.time()
output = m(x98, x93)
end = time.time()
print(end-start)
