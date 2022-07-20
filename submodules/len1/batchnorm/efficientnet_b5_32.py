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
        self.batchnorm2d32 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x168):
        x169=self.batchnorm2d32(x168)
        return x169

m = M().eval()
x168 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x168)
end = time.time()
print(end-start)