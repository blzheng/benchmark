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

    def forward(self, x451, x439):
        x452=operator.add(x451, (12, 64))
        x453=x439.view(x452)
        return x453

m = M().eval()
x451 = (1, 384, )
x439 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x451, x439)
end = time.time()
print(end-start)
