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
        self.batchnorm2d6 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = ReLU(inplace=True)

    def forward(self, x18):
        x19=self.batchnorm2d6(x18)
        x20=self.relu4(x19)
        return x20

m = M().eval()
x18 = torch.randn(torch.Size([1, 128, 56, 56]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)