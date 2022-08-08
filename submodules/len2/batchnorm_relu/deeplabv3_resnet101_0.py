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
        self.batchnorm2d0 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)

    def forward(self, x3):
        x4=self.batchnorm2d0(x3)
        x5=self.relu0(x4)
        return x5

m = M().eval()
x3 = torch.randn(torch.Size([1, 64, 112, 112]))
start = time.time()
output = m(x3)
end = time.time()
print(end-start)
