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
        self.conv2d134 = Conv2d(2904, 7392, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d82 = BatchNorm2d(7392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu108 = ReLU(inplace=True)

    def forward(self, x425, x441):
        x426=self.conv2d134(x425)
        x427=self.batchnorm2d82(x426)
        x442=operator.add(x427, x441)
        x443=self.relu108(x442)
        return x443

m = M().eval()
x425 = torch.randn(torch.Size([1, 2904, 14, 14]))
x441 = torch.randn(torch.Size([1, 7392, 7, 7]))
start = time.time()
output = m(x425, x441)
end = time.time()
print(end-start)
