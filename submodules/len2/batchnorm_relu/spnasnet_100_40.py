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
        self.batchnorm2d60 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)

    def forward(self, x196):
        x197=self.batchnorm2d60(x196)
        x198=self.relu40(x197)
        return x198

m = M().eval()
x196 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x196)
end = time.time()
print(end-start)
