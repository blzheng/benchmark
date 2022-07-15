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
        self.batchnorm2d121 = BatchNorm2d(1344, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x581):
        x582=self.batchnorm2d121(x581)
        return x582

m = M().eval()
x581 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x581)
end = time.time()
print(end-start)
