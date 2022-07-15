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
        self.batchnorm2d100 = BatchNorm2d(704, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x356):
        x357=self.batchnorm2d100(x356)
        return x357

m = M().eval()
x356 = torch.randn(torch.Size([1, 704, 7, 7]))
start = time.time()
output = m(x356)
end = time.time()
print(end-start)
