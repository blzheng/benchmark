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
        self.batchnorm2d32 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x105):
        x106=self.batchnorm2d32(x105)
        return x106

m = M().eval()
x105 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x105)
end = time.time()
print(end-start)
