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

    def forward(self, x318, x303):
        x319=operator.add(x318, x303)
        return x319

m = M().eval()
x318 = torch.randn(torch.Size([1, 160, 14, 14]))
x303 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x318, x303)
end = time.time()
print(end-start)
