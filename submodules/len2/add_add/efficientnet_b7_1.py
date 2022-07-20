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

    def forward(self, x146, x131, x162):
        x147=operator.add(x146, x131)
        x163=operator.add(x162, x147)
        return x163

m = M().eval()
x146 = torch.randn(torch.Size([1, 48, 56, 56]))
x131 = torch.randn(torch.Size([1, 48, 56, 56]))
x162 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x146, x131, x162)
end = time.time()
print(end-start)
