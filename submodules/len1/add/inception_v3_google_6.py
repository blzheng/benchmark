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

    def forward(self, x99):
        x112=torch.nn.functional.max_pool2d(x99, 3,stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=False)
        return x112

m = M().eval()
x99 = torch.randn(torch.Size([1, 288, 25, 25]))
start = time.time()
output = m(x99)
end = time.time()
print(end-start)
