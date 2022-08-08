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
        self.batchnorm2d26 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x96):
        x97=self.batchnorm2d26(x96)
        return x97

m = M().eval()
x96 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x96)
end = time.time()
print(end-start)
