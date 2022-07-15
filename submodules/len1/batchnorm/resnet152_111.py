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
        self.batchnorm2d111 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x367):
        x368=self.batchnorm2d111(x367)
        return x368

m = M().eval()
x367 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x367)
end = time.time()
print(end-start)
