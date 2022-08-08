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

    def forward(self, x111, x122):
        x123=torch.matmul(x111, x122)
        return x123

m = M().eval()
x111 = torch.randn(torch.Size([1, 12, 384, 64]))
x122 = torch.randn(torch.Size([1, 12, 64, 384]))
start = time.time()
output = m(x111, x122)
end = time.time()
print(end-start)
