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
        self.conv2d35 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(128, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x100):
        x101=self.conv2d35(x100)
        x102=self.batchnorm2d35(x101)
        return x102

m = M().eval()
x100 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x100)
end = time.time()
print(end-start)
