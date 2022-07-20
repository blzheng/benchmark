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
        self.batchnorm2d3 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x25):
        x26=self.batchnorm2d3(x25)
        x27=torch.nn.functional.relu(x26,inplace=True)
        return x27

m = M().eval()
x25 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x25)
end = time.time()
print(end-start)
