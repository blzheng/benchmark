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
        self.batchnorm2d53 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x173):
        x174=self.batchnorm2d53(x173)
        return x174

m = M().eval()
x173 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x173)
end = time.time()
print(end-start)
