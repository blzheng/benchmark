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
        self.batchnorm2d82 = BatchNorm2d(1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x396):
        x397=self.batchnorm2d82(x396)
        return x397

m = M().eval()
x396 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x396)
end = time.time()
print(end-start)
