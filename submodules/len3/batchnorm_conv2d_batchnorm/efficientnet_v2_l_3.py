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
        self.batchnorm2d65 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d88 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(1344, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x288):
        x289=self.batchnorm2d65(x288)
        x290=self.conv2d88(x289)
        x291=self.batchnorm2d66(x290)
        return x291

m = M().eval()
x288 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x288)
end = time.time()
print(end-start)
