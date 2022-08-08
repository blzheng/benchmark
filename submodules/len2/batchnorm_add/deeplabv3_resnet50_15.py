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
        self.batchnorm2d42 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x139, x132):
        x140=self.batchnorm2d42(x139)
        x141=operator.add(x140, x132)
        return x141

m = M().eval()
x139 = torch.randn(torch.Size([1, 1024, 28, 28]))
x132 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x139, x132)
end = time.time()
print(end-start)
