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
        self.batchnorm2d69 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x227):
        x228=self.batchnorm2d69(x227)
        return x228

m = M().eval()
x227 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x227)
end = time.time()
print(end-start)
