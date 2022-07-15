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
        self.batchnorm2d124 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x595):
        x596=self.batchnorm2d124(x595)
        return x596

m = M().eval()
x595 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x595)
end = time.time()
print(end-start)
