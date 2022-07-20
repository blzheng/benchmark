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
        self.batchnorm2d57 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x294, x281):
        x295=self.batchnorm2d57(x294)
        x296=operator.add(x281, x295)
        return x296

m = M().eval()
x294 = torch.randn(torch.Size([1, 336, 14, 14]))
x281 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x294, x281)
end = time.time()
print(end-start)
