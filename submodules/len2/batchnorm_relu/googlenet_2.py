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
        self.batchnorm2d2 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x21):
        x22=self.batchnorm2d2(x21)
        x23=torch.nn.functional.relu(x22,inplace=True)
        return x23

m = M().eval()
x21 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x21)
end = time.time()
print(end-start)
