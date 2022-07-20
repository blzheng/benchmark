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
        self.batchnorm2d134 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu130 = ReLU(inplace=True)

    def forward(self, x444):
        x445=self.batchnorm2d134(x444)
        x446=self.relu130(x445)
        return x446

m = M().eval()
x444 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x444)
end = time.time()
print(end-start)
