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
        self.batchnorm2d45 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x230, x217):
        x231=self.batchnorm2d45(x230)
        x232=operator.add(x217, x231)
        return x232

m = M().eval()
x230 = torch.randn(torch.Size([1, 896, 14, 14]))
x217 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x230, x217)
end = time.time()
print(end-start)
