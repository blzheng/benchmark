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
        self.batchnorm2d42 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu628 = ReLU6(inplace=True)

    def forward(self, x121):
        x122=self.batchnorm2d42(x121)
        x123=self.relu628(x122)
        return x123

m = M().eval()
x121 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x121)
end = time.time()
print(end-start)
