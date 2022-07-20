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
        self.batchnorm2d34 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu23 = ReLU(inplace=True)

    def forward(self, x111):
        x112=self.batchnorm2d34(x111)
        x113=self.relu23(x112)
        return x113

m = M().eval()
x111 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x111)
end = time.time()
print(end-start)
