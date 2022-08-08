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

    def forward(self, x246, x257):
        x258=x246.transpose(-1, -2)
        x259=torch.matmul(x257, x258)
        return x259

m = M().eval()
x246 = torch.randn(torch.Size([1, 4, 384, 64]))
x257 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x246, x257)
end = time.time()
print(end-start)
