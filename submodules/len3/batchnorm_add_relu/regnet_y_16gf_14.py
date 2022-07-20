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
        self.batchnorm2d39 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu48 = ReLU(inplace=True)

    def forward(self, x198, x185):
        x199=self.batchnorm2d39(x198)
        x200=operator.add(x185, x199)
        x201=self.relu48(x200)
        return x201

m = M().eval()
x198 = torch.randn(torch.Size([1, 1232, 14, 14]))
x185 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x198, x185)
end = time.time()
print(end-start)
