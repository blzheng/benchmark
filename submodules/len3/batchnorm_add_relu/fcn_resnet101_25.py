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
        self.batchnorm2d72 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU(inplace=True)

    def forward(self, x239, x232):
        x240=self.batchnorm2d72(x239)
        x241=operator.add(x240, x232)
        x242=self.relu67(x241)
        return x242

m = M().eval()
x239 = torch.randn(torch.Size([1, 1024, 28, 28]))
x232 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x239, x232)
end = time.time()
print(end-start)
