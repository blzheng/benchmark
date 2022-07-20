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
        self.batchnorm2d80 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x409, x396):
        x410=self.batchnorm2d80(x409)
        x411=operator.add(x410, x396)
        return x411

m = M().eval()
x409 = torch.randn(torch.Size([1, 160, 14, 14]))
x396 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x409, x396)
end = time.time()
print(end-start)