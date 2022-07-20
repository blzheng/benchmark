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

    def forward(self, x778, x763, x794):
        x779=operator.add(x778, x763)
        x795=operator.add(x794, x779)
        return x795

m = M().eval()
x778 = torch.randn(torch.Size([1, 384, 7, 7]))
x763 = torch.randn(torch.Size([1, 384, 7, 7]))
x794 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x778, x763, x794)
end = time.time()
print(end-start)
