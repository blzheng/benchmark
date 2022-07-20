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
        self.batchnorm2d72 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu69 = ReLU(inplace=True)

    def forward(self, x236, x229):
        x237=self.batchnorm2d72(x236)
        x238=operator.add(x229, x237)
        x239=self.relu69(x238)
        return x239

m = M().eval()
x236 = torch.randn(torch.Size([1, 432, 14, 14]))
x229 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x236, x229)
end = time.time()
print(end-start)
