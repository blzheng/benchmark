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
        self.batchnorm2d28 = BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x112):
        x113=self.batchnorm2d28(x112)
        return x113

m = M().eval()
x112 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x112)
end = time.time()
print(end-start)
