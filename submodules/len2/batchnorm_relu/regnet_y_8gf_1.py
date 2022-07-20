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
        self.batchnorm2d2 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)

    def forward(self, x6):
        x7=self.batchnorm2d2(x6)
        x8=self.relu1(x7)
        return x8

m = M().eval()
x6 = torch.randn(torch.Size([1, 224, 112, 112]))
start = time.time()
output = m(x6)
end = time.time()
print(end-start)
