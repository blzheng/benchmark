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
        self.batchnorm2d36 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)

    def forward(self, x116, x109):
        x117=self.batchnorm2d36(x116)
        x118=operator.add(x109, x117)
        x119=self.relu33(x118)
        return x119

m = M().eval()
x116 = torch.randn(torch.Size([1, 720, 14, 14]))
x109 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x116, x109)
end = time.time()
print(end-start)
