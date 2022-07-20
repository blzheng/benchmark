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
        self.batchnorm2d71 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x232):
        x233=self.batchnorm2d71(x232)
        return x233

m = M().eval()
x232 = torch.randn(torch.Size([1, 1920, 14, 14]))
start = time.time()
output = m(x232)
end = time.time()
print(end-start)