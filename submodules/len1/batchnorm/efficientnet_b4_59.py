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
        self.batchnorm2d59 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x308):
        x309=self.batchnorm2d59(x308)
        return x309

m = M().eval()
x308 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x308)
end = time.time()
print(end-start)
