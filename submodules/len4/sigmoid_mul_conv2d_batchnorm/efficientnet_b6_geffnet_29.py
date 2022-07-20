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
        self.conv2d147 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d87 = BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x437, x433):
        x438=x437.sigmoid()
        x439=operator.mul(x433, x438)
        x440=self.conv2d147(x439)
        x441=self.batchnorm2d87(x440)
        return x441

m = M().eval()
x437 = torch.randn(torch.Size([1, 1200, 1, 1]))
x433 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x437, x433)
end = time.time()
print(end-start)
