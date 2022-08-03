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

    def forward(self, x161, x172):
        x173=x161.transpose(-1, -2)
        x174=torch.matmul(x172, x173)
        return x174

m = M().eval()
x161 = torch.randn(torch.Size([1, 12, 384, 64]))
x172 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x161, x172)
end = time.time()
print(end-start)
