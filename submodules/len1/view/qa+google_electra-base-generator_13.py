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

    def forward(self, x163, x166):
        x167=x163.view(x166)
        return x167

m = M().eval()
x163 = torch.randn(torch.Size([1, 384, 256]))
x166 = (1, 384, 4, 64, )
start = time.time()
output = m(x163, x166)
end = time.time()
print(end-start)
