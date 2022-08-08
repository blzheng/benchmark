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

    def forward(self, x36, x47):
        x48=x36.transpose(-1, -2)
        x49=torch.matmul(x47, x48)
        return x49

m = M().eval()
x36 = torch.randn(torch.Size([1, 4, 384, 64]))
x47 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x36, x47)
end = time.time()
print(end-start)
