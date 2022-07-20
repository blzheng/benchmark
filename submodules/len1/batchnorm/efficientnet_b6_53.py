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
        self.batchnorm2d53 = BatchNorm2d(864, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x278):
        x279=self.batchnorm2d53(x278)
        return x279

m = M().eval()
x278 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x278)
end = time.time()
print(end-start)