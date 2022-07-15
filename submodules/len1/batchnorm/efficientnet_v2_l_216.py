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
        self.batchnorm2d216 = BatchNorm2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x1086):
        x1087=self.batchnorm2d216(x1086)
        return x1087

m = M().eval()
x1086 = torch.randn(torch.Size([1, 1280, 7, 7]))
start = time.time()
output = m(x1086)
end = time.time()
print(end-start)
