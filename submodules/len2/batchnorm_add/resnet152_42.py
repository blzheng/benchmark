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
        self.batchnorm2d123 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x407, x400):
        x408=self.batchnorm2d123(x407)
        x409=operator.add(x408, x400)
        return x409

m = M().eval()
x407 = torch.randn(torch.Size([1, 1024, 14, 14]))
x400 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x407, x400)
end = time.time()
print(end-start)
