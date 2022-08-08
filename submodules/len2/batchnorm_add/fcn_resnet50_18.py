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
        self.batchnorm2d49 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x161, x154):
        x162=self.batchnorm2d49(x161)
        x163=operator.add(x162, x154)
        return x163

m = M().eval()
x161 = torch.randn(torch.Size([1, 2048, 28, 28]))
x154 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x161, x154)
end = time.time()
print(end-start)
