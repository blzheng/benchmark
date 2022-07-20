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
        self.conv2d37 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x117, x111):
        x118=self.conv2d37(x117)
        x119=self.batchnorm2d37(x118)
        x120=operator.add(x111, x119)
        return x120

m = M().eval()
x117 = torch.randn(torch.Size([1, 400, 7, 7]))
x111 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x117, x111)
end = time.time()
print(end-start)
