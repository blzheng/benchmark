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
        self.batchnorm2d127 = BatchNorm2d(1664, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu127 = ReLU(inplace=True)

    def forward(self, x449):
        x450=self.batchnorm2d127(x449)
        x451=self.relu127(x450)
        return x451

m = M().eval()
x449 = torch.randn(torch.Size([1, 1664, 14, 14]))
start = time.time()
output = m(x449)
end = time.time()
print(end-start)
