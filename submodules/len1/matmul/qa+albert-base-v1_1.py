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

    def forward(self, x54, x47):
        x55=torch.matmul(x54, x47)
        return x55

m = M().eval()
x54 = torch.randn(torch.Size([1, 12, 384, 384]))
x47 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x54, x47)
end = time.time()
print(end-start)
