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

    def forward(self, x165, x163):
        x166=operator.add(x165, (4, 64))
        x167=x163.view(x166)
        return x167

m = M().eval()
x165 = (1, 384, )
x163 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x165, x163)
end = time.time()
print(end-start)
