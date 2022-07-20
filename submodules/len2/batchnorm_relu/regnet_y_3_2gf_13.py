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
        self.batchnorm2d21 = BatchNorm2d(216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)

    def forward(self, x104):
        x105=self.batchnorm2d21(x104)
        x106=self.relu25(x105)
        return x106

m = M().eval()
x104 = torch.randn(torch.Size([1, 216, 28, 28]))
start = time.time()
output = m(x104)
end = time.time()
print(end-start)
