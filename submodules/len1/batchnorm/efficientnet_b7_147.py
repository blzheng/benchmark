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
        self.batchnorm2d147 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x780):
        x781=self.batchnorm2d147(x780)
        return x781

m = M().eval()
x780 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x780)
end = time.time()
print(end-start)
