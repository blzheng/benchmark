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
        self.batchnorm2d40 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x203):
        x204=self.batchnorm2d40(x203)
        return x204

m = M().eval()
x203 = torch.randn(torch.Size([1, 136, 14, 14]))
start = time.time()
output = m(x203)
end = time.time()
print(end-start)
