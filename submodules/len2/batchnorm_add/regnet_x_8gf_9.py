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
        self.batchnorm2d24 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x78, x87):
        x79=self.batchnorm2d24(x78)
        x88=operator.add(x79, x87)
        return x88

m = M().eval()
x78 = torch.randn(torch.Size([1, 720, 14, 14]))
x87 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x78, x87)
end = time.time()
print(end-start)
