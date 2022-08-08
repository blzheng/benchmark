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
        self.conv2d87 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d87 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu82 = ReLU(inplace=True)

    def forward(self, x288, x282):
        x289=self.conv2d87(x288)
        x290=self.batchnorm2d87(x289)
        x291=operator.add(x290, x282)
        x292=self.relu82(x291)
        return x292

m = M().eval()
x288 = torch.randn(torch.Size([1, 256, 28, 28]))
x282 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x288, x282)
end = time.time()
print(end-start)
