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
        self.batchnorm2d23 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x83):
        x84=self.batchnorm2d23(x83)
        return x84

m = M().eval()
x83 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x83)
end = time.time()
print(end-start)
