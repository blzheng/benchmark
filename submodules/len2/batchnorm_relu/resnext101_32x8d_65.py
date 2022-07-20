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
        self.batchnorm2d101 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu97 = ReLU(inplace=True)

    def forward(self, x333):
        x334=self.batchnorm2d101(x333)
        x335=self.relu97(x334)
        return x335

m = M().eval()
x333 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x333)
end = time.time()
print(end-start)
