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
        self.batchnorm2d71 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x236):
        x237=self.batchnorm2d71(x236)
        return x237

m = M().eval()
x236 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x236)
end = time.time()
print(end-start)
