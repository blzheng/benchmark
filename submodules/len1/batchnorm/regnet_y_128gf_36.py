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
        self.batchnorm2d36 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x182):
        x183=self.batchnorm2d36(x182)
        return x183

m = M().eval()
x182 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x182)
end = time.time()
print(end-start)
