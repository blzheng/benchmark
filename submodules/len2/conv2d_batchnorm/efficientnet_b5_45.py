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
        self.conv2d77 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x238):
        x239=self.conv2d77(x238)
        x240=self.batchnorm2d45(x239)
        return x240

m = M().eval()
x238 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x238)
end = time.time()
print(end-start)
