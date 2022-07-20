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
        self.batchnorm2d57 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x186, x179):
        x187=self.batchnorm2d57(x186)
        x188=operator.add(x179, x187)
        return x188

m = M().eval()
x186 = torch.randn(torch.Size([1, 432, 14, 14]))
x179 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x186, x179)
end = time.time()
print(end-start)
