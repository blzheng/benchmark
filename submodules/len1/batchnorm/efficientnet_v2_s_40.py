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
        self.batchnorm2d40 = BatchNorm2d(960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x174):
        x175=self.batchnorm2d40(x174)
        return x175

m = M().eval()
x174 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
