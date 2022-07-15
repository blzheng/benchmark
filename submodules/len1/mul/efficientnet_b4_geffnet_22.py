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

    def forward(self, x330, x335):
        x336=operator.mul(x330, x335)
        return x336

m = M().eval()
x330 = torch.randn(torch.Size([1, 960, 7, 7]))
x335 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x330, x335)
end = time.time()
print(end-start)
