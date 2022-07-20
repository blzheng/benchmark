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
        self.batchnorm2d11 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x57, x72):
        x58=self.batchnorm2d11(x57)
        x73=operator.add(x72, x58)
        return x73

m = M().eval()
x57 = torch.randn(torch.Size([1, 40, 28, 28]))
x72 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x57, x72)
end = time.time()
print(end-start)
