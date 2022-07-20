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
        self.batchnorm2d37 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x138):
        x139=self.batchnorm2d37(x138)
        x140=torch.nn.functional.relu(x139,inplace=True)
        return x140

m = M().eval()
x138 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x138)
end = time.time()
print(end-start)
