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
        self.conv2d147 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x476, x471):
        x477=operator.mul(x476, x471)
        x478=self.conv2d147(x477)
        x479=self.batchnorm2d101(x478)
        return x479

m = M().eval()
x476 = torch.randn(torch.Size([1, 1344, 1, 1]))
x471 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x476, x471)
end = time.time()
print(end-start)
