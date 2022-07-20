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
        self.batchnorm2d4 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)

    def forward(self, x13, x12):
        x14=self.batchnorm2d4(x13)
        x15=operator.add(x12, x14)
        x16=self.relu1(x15)
        return x16

m = M().eval()
x13 = torch.randn(torch.Size([1, 256, 56, 56]))
x12 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x13, x12)
end = time.time()
print(end-start)
