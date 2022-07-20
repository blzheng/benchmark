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
        self.batchnorm2d87 = BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu87 = ReLU(inplace=True)

    def forward(self, x309):
        x310=self.batchnorm2d87(x309)
        x311=self.relu87(x310)
        return x311

m = M().eval()
x309 = torch.randn(torch.Size([1, 1536, 14, 14]))
start = time.time()
output = m(x309)
end = time.time()
print(end-start)
