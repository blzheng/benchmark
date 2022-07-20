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
        self.batchnorm2d24 = BatchNorm2d(384, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)

    def forward(self, x69):
        x70=self.batchnorm2d24(x69)
        x71=self.relu16(x70)
        return x71

m = M().eval()
x69 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x69)
end = time.time()
print(end-start)
