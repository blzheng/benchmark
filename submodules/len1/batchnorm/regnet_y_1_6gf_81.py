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
        self.batchnorm2d81 = BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x415):
        x416=self.batchnorm2d81(x415)
        return x416

m = M().eval()
x415 = torch.randn(torch.Size([1, 888, 7, 7]))
start = time.time()
output = m(x415)
end = time.time()
print(end-start)
