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
        self.batchnorm2d43 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x224):
        x225=self.batchnorm2d43(x224)
        return x225

m = M().eval()
x224 = torch.randn(torch.Size([1, 120, 14, 14]))
start = time.time()
output = m(x224)
end = time.time()
print(end-start)
