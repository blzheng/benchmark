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

    def forward(self, x307, x293):
        x308=operator.add(x307, x293)
        return x308

m = M().eval()
x307 = torch.randn(torch.Size([1, 192, 7, 7]))
x293 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x307, x293)
end = time.time()
print(end-start)
