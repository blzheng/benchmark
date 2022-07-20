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
        self.batchnorm2d59 = BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu59 = ReLU(inplace=True)

    def forward(self, x211):
        x212=self.batchnorm2d59(x211)
        x213=self.relu59(x212)
        return x213

m = M().eval()
x211 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x211)
end = time.time()
print(end-start)
