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
        self.batchnorm2d37 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x188, x175):
        x189=self.batchnorm2d37(x188)
        x190=operator.add(x189, x175)
        return x190

m = M().eval()
x188 = torch.randn(torch.Size([1, 96, 14, 14]))
x175 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x188, x175)
end = time.time()
print(end-start)
