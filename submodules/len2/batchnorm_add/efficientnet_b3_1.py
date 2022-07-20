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
        self.batchnorm2d73 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x382, x398):
        x383=self.batchnorm2d73(x382)
        x399=operator.add(x398, x383)
        return x399

m = M().eval()
x382 = torch.randn(torch.Size([1, 384, 7, 7]))
x398 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x382, x398)
end = time.time()
print(end-start)
