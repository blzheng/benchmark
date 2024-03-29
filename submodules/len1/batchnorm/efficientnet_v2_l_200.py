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
        self.batchnorm2d200 = BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x1002):
        x1003=self.batchnorm2d200(x1002)
        return x1003

m = M().eval()
x1002 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x1002)
end = time.time()
print(end-start)
