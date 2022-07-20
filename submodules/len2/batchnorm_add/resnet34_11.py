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
        self.batchnorm2d22 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x75, x71):
        x76=self.batchnorm2d22(x75)
        x77=operator.add(x76, x71)
        return x77

m = M().eval()
x75 = torch.randn(torch.Size([1, 256, 14, 14]))
x71 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x75, x71)
end = time.time()
print(end-start)
