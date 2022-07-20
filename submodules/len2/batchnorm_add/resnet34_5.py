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
        self.batchnorm2d11 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x38, x34):
        x39=self.batchnorm2d11(x38)
        x40=operator.add(x39, x34)
        return x40

m = M().eval()
x38 = torch.randn(torch.Size([1, 128, 28, 28]))
x34 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x38, x34)
end = time.time()
print(end-start)
