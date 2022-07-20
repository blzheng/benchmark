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
        self.batchnorm2d22 = BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x89):
        x90=self.batchnorm2d22(x89)
        x91=torch.nn.functional.relu(x90,inplace=True)
        return x91

m = M().eval()
x89 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x89)
end = time.time()
print(end-start)
