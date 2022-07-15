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

    def forward(self, x557, x542):
        x558=operator.add(x557, x542)
        return x558

m = M().eval()
x557 = torch.randn(torch.Size([1, 304, 7, 7]))
x542 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x557, x542)
end = time.time()
print(end-start)
