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
        self.batchnorm2d31 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x165):
        x166=self.batchnorm2d31(x165)
        return x166

m = M().eval()
x165 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x165)
end = time.time()
print(end-start)