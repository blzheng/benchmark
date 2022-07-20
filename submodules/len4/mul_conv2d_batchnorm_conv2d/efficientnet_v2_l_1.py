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
        self.conv2d87 = Conv2d(1152, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d65 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d88 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x286, x281):
        x287=operator.mul(x286, x281)
        x288=self.conv2d87(x287)
        x289=self.batchnorm2d65(x288)
        x290=self.conv2d88(x289)
        return x290

m = M().eval()
x286 = torch.randn(torch.Size([1, 1152, 1, 1]))
x281 = torch.randn(torch.Size([1, 1152, 14, 14]))
start = time.time()
output = m(x286, x281)
end = time.time()
print(end-start)
