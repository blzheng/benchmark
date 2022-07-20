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
        self.batchnorm2d105 = BatchNorm2d(1968, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu105 = ReLU(inplace=True)

    def forward(self, x372):
        x373=self.batchnorm2d105(x372)
        x374=self.relu105(x373)
        return x374

m = M().eval()
x372 = torch.randn(torch.Size([1, 1968, 14, 14]))
start = time.time()
output = m(x372)
end = time.time()
print(end-start)
