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
        self.batchnorm2d6 = BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x37):
        x38=self.batchnorm2d6(x37)
        return x38

m = M().eval()
x37 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
