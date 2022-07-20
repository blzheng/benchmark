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
        self.batchnorm2d29 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x94, x87):
        x95=self.batchnorm2d29(x94)
        x96=operator.add(x87, x95)
        return x96

m = M().eval()
x94 = torch.randn(torch.Size([1, 672, 28, 28]))
x87 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x94, x87)
end = time.time()
print(end-start)