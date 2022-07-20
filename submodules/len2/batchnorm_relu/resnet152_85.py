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
        self.batchnorm2d130 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu127 = ReLU(inplace=True)

    def forward(self, x431):
        x432=self.batchnorm2d130(x431)
        x433=self.relu127(x432)
        return x433

m = M().eval()
x431 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x431)
end = time.time()
print(end-start)
