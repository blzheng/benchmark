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

    def forward(self, x299, x300):
        x301=torch.matmul(x299, x300)
        return x301

m = M().eval()
x299 = torch.randn(torch.Size([1, 4, 384, 64]))
x300 = torch.randn(torch.Size([1, 4, 64, 384]))
start = time.time()
output = m(x299, x300)
end = time.time()
print(end-start)
