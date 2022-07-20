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
        self.batchnorm2d51 = BatchNorm2d(488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)

    def forward(self, x327):
        x328=self.batchnorm2d51(x327)
        x329=self.relu33(x328)
        return x329

m = M().eval()
x327 = torch.randn(torch.Size([1, 488, 7, 7]))
start = time.time()
output = m(x327)
end = time.time()
print(end-start)
