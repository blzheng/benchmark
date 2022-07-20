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
        self.batchnorm2d85 = BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x293):
        x294=self.batchnorm2d85(x293)
        x295=torch.nn.functional.relu(x294,inplace=True)
        return x295

m = M().eval()
x293 = torch.randn(torch.Size([1, 320, 5, 5]))
start = time.time()
output = m(x293)
end = time.time()
print(end-start)
