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
        self.conv2d283 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d183 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x910, x895):
        x911=operator.add(x910, x895)
        x912=self.conv2d283(x911)
        x913=self.batchnorm2d183(x912)
        return x913

m = M().eval()
x910 = torch.randn(torch.Size([1, 384, 7, 7]))
x895 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x910, x895)
end = time.time()
print(end-start)
