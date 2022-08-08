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

    def forward(self, x88, x89):
        x90=torch.matmul(x88, x89)
        return x90

m = M().eval()
x88 = torch.randn(torch.Size([1, 12, 384, 64]))
x89 = torch.randn(torch.Size([1, 12, 64, 384]))
start = time.time()
output = m(x88, x89)
end = time.time()
print(end-start)
