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
        self.batchnorm2d120 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x397, x390):
        x398=self.batchnorm2d120(x397)
        x399=operator.add(x398, x390)
        return x399

m = M().eval()
x397 = torch.randn(torch.Size([1, 1024, 14, 14]))
x390 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x397, x390)
end = time.time()
print(end-start)
