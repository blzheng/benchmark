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

    def forward(self, x340, x341):
        x342=torch.matmul(x340, x341)
        return x342

m = M().eval()
x340 = torch.randn(torch.Size([1, 12, 384, 64]))
x341 = torch.randn(torch.Size([1, 12, 64, 384]))
start = time.time()
output = m(x340, x341)
end = time.time()
print(end-start)
