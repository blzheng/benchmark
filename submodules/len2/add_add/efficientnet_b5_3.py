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

    def forward(self, x289, x274, x305):
        x290=operator.add(x289, x274)
        x306=operator.add(x305, x290)
        return x306

m = M().eval()
x289 = torch.randn(torch.Size([1, 128, 14, 14]))
x274 = torch.randn(torch.Size([1, 128, 14, 14]))
x305 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x289, x274, x305)
end = time.time()
print(end-start)
