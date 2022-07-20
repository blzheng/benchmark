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
        self.batchnorm2d130 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu130 = ReLU(inplace=True)

    def forward(self, x459):
        x460=self.batchnorm2d130(x459)
        x461=self.relu130(x460)
        return x461

m = M().eval()
x459 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x459)
end = time.time()
print(end-start)
