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
        self.batchnorm2d59 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x301):
        x302=self.batchnorm2d59(x301)
        return x302

m = M().eval()
x301 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x301)
end = time.time()
print(end-start)
