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
        self.batchnorm2d18 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)

    def forward(self, x88):
        x89=self.batchnorm2d18(x88)
        x90=self.relu21(x89)
        return x90

m = M().eval()
x88 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x88)
end = time.time()
print(end-start)
