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
        self.batchnorm2d162 = BatchNorm2d(2560, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x858):
        x859=self.batchnorm2d162(x858)
        return x859

m = M().eval()
x858 = torch.randn(torch.Size([1, 2560, 7, 7]))
start = time.time()
output = m(x858)
end = time.time()
print(end-start)
