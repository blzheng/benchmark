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
        self.batchnorm2d51 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x187):
        x188=self.batchnorm2d51(x187)
        x189=torch.nn.functional.relu(x188,inplace=True)
        return x189

m = M().eval()
x187 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x187)
end = time.time()
print(end-start)
