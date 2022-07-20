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
        self.batchnorm2d52 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x168, x161):
        x169=self.batchnorm2d52(x168)
        x170=operator.add(x161, x169)
        return x170

m = M().eval()
x168 = torch.randn(torch.Size([1, 672, 7, 7]))
x161 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x168, x161)
end = time.time()
print(end-start)
