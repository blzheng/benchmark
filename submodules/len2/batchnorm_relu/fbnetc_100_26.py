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
        self.batchnorm2d38 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)

    def forward(self, x124):
        x125=self.batchnorm2d38(x124)
        x126=self.relu26(x125)
        return x126

m = M().eval()
x124 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x124)
end = time.time()
print(end-start)