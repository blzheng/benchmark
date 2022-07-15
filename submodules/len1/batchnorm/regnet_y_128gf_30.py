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
        self.batchnorm2d30 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x152):
        x153=self.batchnorm2d30(x152)
        return x153

m = M().eval()
x152 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
