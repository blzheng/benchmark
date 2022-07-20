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
        self.batchnorm2d49 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x158, x151):
        x159=self.batchnorm2d49(x158)
        x160=operator.add(x151, x159)
        return x160

m = M().eval()
x158 = torch.randn(torch.Size([1, 400, 7, 7]))
x151 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x158, x151)
end = time.time()
print(end-start)
