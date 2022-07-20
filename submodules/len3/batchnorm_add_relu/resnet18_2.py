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
        self.batchnorm2d6 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)

    def forward(self, x22, x25):
        x23=self.batchnorm2d6(x22)
        x26=operator.add(x23, x25)
        x27=self.relu5(x26)
        return x27

m = M().eval()
x22 = torch.randn(torch.Size([1, 128, 28, 28]))
x25 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x22, x25)
end = time.time()
print(end-start)
