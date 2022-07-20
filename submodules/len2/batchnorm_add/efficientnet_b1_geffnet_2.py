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
        self.batchnorm2d34 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x173, x160):
        x174=self.batchnorm2d34(x173)
        x175=operator.add(x174, x160)
        return x175

m = M().eval()
x173 = torch.randn(torch.Size([1, 80, 14, 14]))
x160 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x173, x160)
end = time.time()
print(end-start)
