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

    def forward(self, x451, x447):
        x452=x451.sigmoid()
        x453=operator.mul(x447, x452)
        return x453

m = M().eval()
x451 = torch.randn(torch.Size([1, 1824, 1, 1]))
x447 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x451, x447)
end = time.time()
print(end-start)
