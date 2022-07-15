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

    def forward(self, x477, x462):
        x478=operator.add(x477, x462)
        return x478

m = M().eval()
x477 = torch.randn(torch.Size([1, 304, 7, 7]))
x462 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x477, x462)
end = time.time()
print(end-start)
