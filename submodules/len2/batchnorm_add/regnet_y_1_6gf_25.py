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
        self.batchnorm2d72 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x374, x361):
        x375=self.batchnorm2d72(x374)
        x376=operator.add(x361, x375)
        return x376

m = M().eval()
x374 = torch.randn(torch.Size([1, 336, 14, 14]))
x361 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x374, x361)
end = time.time()
print(end-start)
