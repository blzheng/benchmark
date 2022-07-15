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
        self.batchnorm2d144 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x704):
        x705=self.batchnorm2d144(x704)
        return x705

m = M().eval()
x704 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x704)
end = time.time()
print(end-start)
