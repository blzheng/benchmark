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

    def forward(self, x544, x530, x559):
        x545=operator.add(x544, x530)
        x560=operator.add(x559, x545)
        return x560

m = M().eval()
x544 = torch.randn(torch.Size([1, 224, 14, 14]))
x530 = torch.randn(torch.Size([1, 224, 14, 14]))
x559 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x544, x530, x559)
end = time.time()
print(end-start)
