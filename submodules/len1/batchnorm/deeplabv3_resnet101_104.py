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
        self.batchnorm2d104 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x345):
        x346=self.batchnorm2d104(x345)
        return x346

m = M().eval()
x345 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x345)
end = time.time()
print(end-start)
