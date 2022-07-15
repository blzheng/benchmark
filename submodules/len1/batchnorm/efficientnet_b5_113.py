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
        self.batchnorm2d113 = BatchNorm2d(3072, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x592):
        x593=self.batchnorm2d113(x592)
        return x593

m = M().eval()
x592 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x592)
end = time.time()
print(end-start)
