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

    def forward(self, x222, x233):
        x234=torch.matmul(x222, x233)
        return x234

m = M().eval()
x222 = torch.randn(torch.Size([1, 12, 384, 64]))
x233 = torch.randn(torch.Size([1, 12, 64, 384]))
start = time.time()
output = m(x222, x233)
end = time.time()
print(end-start)
