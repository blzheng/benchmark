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
        self.batchnorm2d7 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x25):
        x26=self.batchnorm2d7(x25)
        return x26

m = M().eval()
x25 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x25)
end = time.time()
print(end-start)
