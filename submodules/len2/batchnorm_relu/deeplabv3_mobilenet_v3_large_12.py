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
        self.batchnorm2d47 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU()

    def forward(self, x187):
        x188=self.batchnorm2d47(x187)
        x189=self.relu20(x188)
        return x189

m = M().eval()
x187 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x187)
end = time.time()
print(end-start)
