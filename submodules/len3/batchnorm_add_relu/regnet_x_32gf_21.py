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
        self.batchnorm2d60 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu57 = ReLU(inplace=True)

    def forward(self, x196, x189):
        x197=self.batchnorm2d60(x196)
        x198=operator.add(x189, x197)
        x199=self.relu57(x198)
        return x199

m = M().eval()
x196 = torch.randn(torch.Size([1, 1344, 14, 14]))
x189 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x196, x189)
end = time.time()
print(end-start)
