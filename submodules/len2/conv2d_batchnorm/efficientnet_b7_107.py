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
        self.conv2d181 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d107 = BatchNorm2d(224, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x569):
        x570=self.conv2d181(x569)
        x571=self.batchnorm2d107(x570)
        return x571

m = M().eval()
x569 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x569)
end = time.time()
print(end-start)
