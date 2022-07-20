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
        self.batchnorm2d35 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)

    def forward(self, x114):
        x115=self.batchnorm2d35(x114)
        x116=self.relu31(x115)
        return x116

m = M().eval()
x114 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x114)
end = time.time()
print(end-start)
