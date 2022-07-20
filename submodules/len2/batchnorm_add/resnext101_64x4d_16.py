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
        self.batchnorm2d45 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x147, x140):
        x148=self.batchnorm2d45(x147)
        x149=operator.add(x148, x140)
        return x149

m = M().eval()
x147 = torch.randn(torch.Size([1, 1024, 14, 14]))
x140 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x147, x140)
end = time.time()
print(end-start)
