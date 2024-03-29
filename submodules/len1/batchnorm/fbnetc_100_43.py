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
        self.batchnorm2d43 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x140):
        x141=self.batchnorm2d43(x140)
        return x141

m = M().eval()
x140 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x140)
end = time.time()
print(end-start)
