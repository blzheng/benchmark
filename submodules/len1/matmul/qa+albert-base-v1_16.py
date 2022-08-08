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

    def forward(self, x333, x344):
        x345=torch.matmul(x333, x344)
        return x345

m = M().eval()
x333 = torch.randn(torch.Size([1, 12, 384, 64]))
x344 = torch.randn(torch.Size([1, 12, 64, 384]))
start = time.time()
output = m(x333, x344)
end = time.time()
print(end-start)
