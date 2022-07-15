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
        self.batchnorm2d153 = BatchNorm2d(3840, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x810):
        x811=self.batchnorm2d153(x810)
        return x811

m = M().eval()
x810 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x810)
end = time.time()
print(end-start)
