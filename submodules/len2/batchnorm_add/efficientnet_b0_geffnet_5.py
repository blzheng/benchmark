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
        self.batchnorm2d32 = BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x159, x146):
        x160=self.batchnorm2d32(x159)
        x161=operator.add(x160, x146)
        return x161

m = M().eval()
x159 = torch.randn(torch.Size([1, 112, 14, 14]))
x146 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x159, x146)
end = time.time()
print(end-start)