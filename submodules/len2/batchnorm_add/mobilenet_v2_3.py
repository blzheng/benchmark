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
        self.batchnorm2d29 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x84, x77):
        x85=self.batchnorm2d29(x84)
        x86=operator.add(x77, x85)
        return x86

m = M().eval()
x84 = torch.randn(torch.Size([1, 64, 14, 14]))
x77 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x84, x77)
end = time.time()
print(end-start)
