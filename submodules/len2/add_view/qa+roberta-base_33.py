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

    def forward(self, x374, x372):
        x375=operator.add(x374, (12, 64))
        x376=x372.view(x375)
        return x376

m = M().eval()
x374 = (1, 384, )
x372 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x374, x372)
end = time.time()
print(end-start)
