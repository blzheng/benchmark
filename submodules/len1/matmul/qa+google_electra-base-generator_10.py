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

    def forward(self, x257, x258):
        x259=torch.matmul(x257, x258)
        return x259

m = M().eval()
x257 = torch.randn(torch.Size([1, 4, 384, 64]))
x258 = torch.randn(torch.Size([1, 4, 64, 384]))
start = time.time()
output = m(x257, x258)
end = time.time()
print(end-start)
