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
        self.batchnorm2d107 = BatchNorm2d(1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x527):
        x528=self.batchnorm2d107(x527)
        return x528

m = M().eval()
x527 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x527)
end = time.time()
print(end-start)