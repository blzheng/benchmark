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
        self.batchnorm2d37 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)

    def forward(self, x121):
        x122=self.batchnorm2d37(x121)
        x123=self.relu25(x122)
        return x123

m = M().eval()
x121 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x121)
end = time.time()
print(end-start)
