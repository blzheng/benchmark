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
        self.batchnorm2d64 = BatchNorm2d(960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x302):
        x303=self.batchnorm2d64(x302)
        return x303

m = M().eval()
x302 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x302)
end = time.time()
print(end-start)
