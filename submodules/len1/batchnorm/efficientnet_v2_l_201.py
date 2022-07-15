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
        self.batchnorm2d201 = BatchNorm2d(3840, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x1006):
        x1007=self.batchnorm2d201(x1006)
        return x1007

m = M().eval()
x1006 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1006)
end = time.time()
print(end-start)
