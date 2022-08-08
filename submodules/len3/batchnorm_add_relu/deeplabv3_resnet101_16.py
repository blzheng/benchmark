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
        self.batchnorm2d45 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)

    def forward(self, x149, x142):
        x150=self.batchnorm2d45(x149)
        x151=operator.add(x150, x142)
        x152=self.relu40(x151)
        return x152

m = M().eval()
x149 = torch.randn(torch.Size([1, 1024, 28, 28]))
x142 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x149, x142)
end = time.time()
print(end-start)
