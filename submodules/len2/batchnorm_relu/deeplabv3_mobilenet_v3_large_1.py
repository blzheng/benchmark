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
        self.batchnorm2d3 = BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)

    def forward(self, x12):
        x13=self.batchnorm2d3(x12)
        x14=self.relu1(x13)
        return x14

m = M().eval()
x12 = torch.randn(torch.Size([1, 64, 112, 112]))
start = time.time()
output = m(x12)
end = time.time()
print(end-start)
