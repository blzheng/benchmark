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
        self.batchnorm2d29 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x156, x143):
        x157=self.batchnorm2d29(x156)
        x158=operator.add(x157, x143)
        return x158

m = M().eval()
x156 = torch.randn(torch.Size([1, 48, 56, 56]))
x143 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x156, x143)
end = time.time()
print(end-start)
