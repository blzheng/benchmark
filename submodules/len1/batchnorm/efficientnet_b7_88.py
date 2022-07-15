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
        self.batchnorm2d88 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x465):
        x466=self.batchnorm2d88(x465)
        return x466

m = M().eval()
x465 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x465)
end = time.time()
print(end-start)
