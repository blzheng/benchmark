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
        self.conv2d25 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x73, x58):
        x74=operator.add(x73, x58)
        x75=self.conv2d25(x74)
        x76=self.batchnorm2d15(x75)
        return x76

m = M().eval()
x73 = torch.randn(torch.Size([1, 40, 28, 28]))
x58 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x73, x58)
end = time.time()
print(end-start)
