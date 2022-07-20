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
        self.batchnorm2d133 = BatchNorm2d(1824, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x658):
        x659=self.batchnorm2d133(x658)
        return x659

m = M().eval()
x658 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x658)
end = time.time()
print(end-start)