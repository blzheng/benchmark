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

    def forward(self, x492, x497):
        x498=operator.mul(x492, x497)
        return x498

m = M().eval()
x492 = torch.randn(torch.Size([1, 1824, 7, 7]))
x497 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x492, x497)
end = time.time()
print(end-start)
