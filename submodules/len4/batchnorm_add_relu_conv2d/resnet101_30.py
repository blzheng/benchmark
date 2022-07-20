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
        self.batchnorm2d87 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu82 = ReLU(inplace=True)
        self.conv2d88 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x287, x280):
        x288=self.batchnorm2d87(x287)
        x289=operator.add(x288, x280)
        x290=self.relu82(x289)
        x291=self.conv2d88(x290)
        return x291

m = M().eval()
x287 = torch.randn(torch.Size([1, 1024, 14, 14]))
x280 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x287, x280)
end = time.time()
print(end-start)
