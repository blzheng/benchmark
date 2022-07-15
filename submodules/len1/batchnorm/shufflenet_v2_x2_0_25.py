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
        self.batchnorm2d25 = BatchNorm2d(244, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x149):
        x150=self.batchnorm2d25(x149)
        return x150

m = M().eval()
x149 = torch.randn(torch.Size([1, 244, 14, 14]))
start = time.time()
output = m(x149)
end = time.time()
print(end-start)
