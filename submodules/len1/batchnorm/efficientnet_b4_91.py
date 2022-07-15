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
        self.batchnorm2d91 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x478):
        x479=self.batchnorm2d91(x478)
        return x479

m = M().eval()
x478 = torch.randn(torch.Size([1, 448, 7, 7]))
start = time.time()
output = m(x478)
end = time.time()
print(end-start)
