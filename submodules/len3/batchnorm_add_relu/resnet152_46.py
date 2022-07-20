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
        self.batchnorm2d135 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu130 = ReLU(inplace=True)

    def forward(self, x447, x440):
        x448=self.batchnorm2d135(x447)
        x449=operator.add(x448, x440)
        x450=self.relu130(x449)
        return x450

m = M().eval()
x447 = torch.randn(torch.Size([1, 1024, 14, 14]))
x440 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x447, x440)
end = time.time()
print(end-start)
