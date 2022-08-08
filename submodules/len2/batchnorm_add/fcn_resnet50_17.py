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
        self.batchnorm2d46 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x151, x150):
        x152=self.batchnorm2d46(x151)
        x153=operator.add(x150, x152)
        return x153

m = M().eval()
x151 = torch.randn(torch.Size([1, 2048, 28, 28]))
x150 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x151, x150)
end = time.time()
print(end-start)
