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
        self.batchnorm2d54 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x176, x169):
        x177=self.batchnorm2d54(x176)
        x178=operator.add(x169, x177)
        return x178

m = M().eval()
x176 = torch.randn(torch.Size([1, 720, 14, 14]))
x169 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x176, x169)
end = time.time()
print(end-start)
