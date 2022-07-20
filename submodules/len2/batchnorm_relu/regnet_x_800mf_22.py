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
        self.batchnorm2d35 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)

    def forward(self, x113):
        x114=self.batchnorm2d35(x113)
        x115=self.relu32(x114)
        return x115

m = M().eval()
x113 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x113)
end = time.time()
print(end-start)
