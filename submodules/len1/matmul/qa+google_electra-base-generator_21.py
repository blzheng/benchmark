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

    def forward(self, x474, x462):
        x475=torch.matmul(x474, x462)
        return x475

m = M().eval()
x474 = torch.randn(torch.Size([1, 4, 384, 384]))
x462 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x474, x462)
end = time.time()
print(end-start)
