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
        self.batchnorm2d69 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu88 = ReLU(inplace=True)

    def forward(self, x358, x345):
        x359=self.batchnorm2d69(x358)
        x360=operator.add(x345, x359)
        x361=self.relu88(x360)
        return x361

m = M().eval()
x358 = torch.randn(torch.Size([1, 336, 14, 14]))
x345 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x358, x345)
end = time.time()
print(end-start)
