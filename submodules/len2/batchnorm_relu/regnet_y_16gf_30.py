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
        self.batchnorm2d47 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)

    def forward(self, x237):
        x238=self.batchnorm2d47(x237)
        x239=self.relu58(x238)
        return x239

m = M().eval()
x237 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x237)
end = time.time()
print(end-start)
