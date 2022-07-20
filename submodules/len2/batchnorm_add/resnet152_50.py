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
        self.batchnorm2d147 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x487, x490):
        x488=self.batchnorm2d147(x487)
        x491=operator.add(x488, x490)
        return x491

m = M().eval()
x487 = torch.randn(torch.Size([1, 2048, 7, 7]))
x490 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x487, x490)
end = time.time()
print(end-start)
