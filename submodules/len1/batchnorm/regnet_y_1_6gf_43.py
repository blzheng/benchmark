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
        self.batchnorm2d43 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x218):
        x219=self.batchnorm2d43(x218)
        return x219

m = M().eval()
x218 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x218)
end = time.time()
print(end-start)
