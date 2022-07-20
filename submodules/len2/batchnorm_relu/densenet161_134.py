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
        self.batchnorm2d134 = BatchNorm2d(1584, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu134 = ReLU(inplace=True)

    def forward(self, x475):
        x476=self.batchnorm2d134(x475)
        x477=self.relu134(x476)
        return x477

m = M().eval()
x475 = torch.randn(torch.Size([1, 1584, 7, 7]))
start = time.time()
output = m(x475)
end = time.time()
print(end-start)
