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
        self.batchnorm2d20 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)

    def forward(self, x67, x60):
        x68=self.batchnorm2d20(x67)
        x69=operator.add(x68, x60)
        x70=self.relu16(x69)
        return x70

m = M().eval()
x67 = torch.randn(torch.Size([1, 512, 28, 28]))
x60 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x67, x60)
end = time.time()
print(end-start)
