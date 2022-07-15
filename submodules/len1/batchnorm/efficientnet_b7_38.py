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
        self.batchnorm2d38 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x206):
        x207=self.batchnorm2d38(x206)
        return x207

m = M().eval()
x206 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x206)
end = time.time()
print(end-start)
