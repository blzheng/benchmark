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
        self.batchnorm2d154 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu148 = ReLU(inplace=True)

    def forward(self, x509, x502):
        x510=self.batchnorm2d154(x509)
        x511=operator.add(x510, x502)
        x512=self.relu148(x511)
        return x512

m = M().eval()
x509 = torch.randn(torch.Size([1, 2048, 7, 7]))
x502 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x509, x502)
end = time.time()
print(end-start)
