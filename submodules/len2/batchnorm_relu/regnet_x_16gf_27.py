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
        self.batchnorm2d43 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)

    def forward(self, x140):
        x141=self.batchnorm2d43(x140)
        x142=self.relu40(x141)
        return x142

m = M().eval()
x140 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x140)
end = time.time()
print(end-start)
