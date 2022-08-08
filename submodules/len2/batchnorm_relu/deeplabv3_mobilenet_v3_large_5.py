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
        self.batchnorm2d9 = BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)

    def forward(self, x29):
        x30=self.batchnorm2d9(x29)
        x31=self.relu5(x30)
        return x31

m = M().eval()
x29 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
