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
        self.batchnorm2d68 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x234):
        x235=self.batchnorm2d68(x234)
        x236=torch.nn.functional.relu(x235,inplace=True)
        return x236

m = M().eval()
x234 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x234)
end = time.time()
print(end-start)
