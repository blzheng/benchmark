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
        self.conv2d94 = Conv2d(208, 1248, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x277, x263):
        x278=operator.add(x277, x263)
        x279=self.conv2d94(x278)
        x280=self.batchnorm2d56(x279)
        return x280

m = M().eval()
x277 = torch.randn(torch.Size([1, 208, 7, 7]))
x263 = torch.randn(torch.Size([1, 208, 7, 7]))
start = time.time()
output = m(x277, x263)
end = time.time()
print(end-start)
