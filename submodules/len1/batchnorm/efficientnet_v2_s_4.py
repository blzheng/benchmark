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
        self.batchnorm2d4 = BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x17):
        x18=self.batchnorm2d4(x17)
        return x18

m = M().eval()
x17 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x17)
end = time.time()
print(end-start)